"""Shared six-DOF ascent continuation and supplied Earth-Mars Lambert cases.

``integrated_tmi`` continues the ascent through coast, restart, and finite TMI.
``lambert_window`` compares supplied ephemeris cases. Inputs and result names
include units; C3 uses km^2/s^2. The Lambert solver uses two-body gravity and
finds no global launch-window optimum or ephemeris states of its own.
"""

import copy
import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from .trajectory import Simulation, load_config, run_trajectory, matrix, unit, MU, RE


def _positive(value, name, allow_zero=False):
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}.'
        )
    return float(value)


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain three finite SI values.")
    return value


def state_from_row(row):
    """Reconstruct all 22 integrated states, including attitude and loss integrals."""
    # Keep the trajectory model's state order: position, velocity, quaternion,
    # body rates, two propellant masses, two gimbal angles, and five integrals.
    names = (
        [f"eci_{k}_m" for k in "xyz"]
        + [f"eci_v{k}_m_s" for k in "xyz"]
        + [f"quaternion_{k}" for k in "xyzw"]
    )
    state = [float(row[k]) for k in names]
    state += list(np.deg2rad([row[f"body_rate_{k}_deg_s"] for k in "xyz"]))
    state += [row["prop1_kg"], row["prop2_kg"]]
    state += list(np.deg2rad([row["gimbal_y_deg"], row["gimbal_z_deg"]]))
    state += [
        row[k]
        for k in [
            "ideal_dv_m_s",
            "steering_loss_m_s",
            "gimbal_loss_m_s",
            "drag_loss_m_s",
            "gravity_loss_m_s",
        ]
    ]
    result = np.asarray(state, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Cutoff state contains nonfinite values.")
    return result


class _DepartureSimulation(Simulation):
    def guidance(self, t, r, vrel, v, h, qbar):
        # Prograde steering is an explicit feasible-policy screen, not targeting
        # optimization. The original actuator/6DOF response still follows it.
        return unit(v)


def finite_burn_tmi(
    *,
    cutoff_row,
    config,
    coast_duration_s,
    restart_propellant_kg,
    target_c3_km2_s2,
    max_burn_duration_s,
    output_step_s=1.0,
):
    """Continue an ascent-cutoff state through coast, restart, and finite TMI.

    Required coast/restart values are not silently invented. C3 is the Earth
    two-body characteristic energy, not proof of Mars arrival geometry. Thermal
    boiloff and ignition transients require separate supplied models. The cutoff
    row must describe the detached, fairing-free upper stage; its mass and CG are
    checked against the supplied configuration before any phase transition.
    """
    _positive(coast_duration_s, "coast_duration_s", True)
    _positive(restart_propellant_kg, "restart_propellant_kg", True)
    _positive(max_burn_duration_s, "max_burn_duration_s")
    _positive(output_step_s, "output_step_s")
    if not math.isfinite(target_c3_km2_s2):
        raise ValueError("Target C3 must be finite.")
    run_config = copy.deepcopy(config)
    run_config["solver"]["output_step_s"] = output_step_s
    run_config["solver"]["verbose"] = False
    simulation = _DepartureSimulation(run_config)
    simulation.phase = 3
    simulation.attached = False
    simulation.fairing = False
    simulation.cutoff = True
    state = state_from_row(cutoff_row)
    time = float(cutoff_row["t_s"])
    start_time = time
    if "mass_kg" not in cutoff_row or "cg_from_base_m" not in cutoff_row:
        raise ValueError(
            "Cutoff row must include mass_kg and cg_from_base_m to verify vehicle configuration."
        )
    supplied_mass_cg = np.array(
        [cutoff_row["mass_kg"], cutoff_row["cg_from_base_m"]], dtype=float
    )

    def matches(fairing):
        expected = np.asarray(
            simulation.vehicle.properties(state[13:15], False, fairing)[:2]
        )
        return bool(
            np.all(np.isfinite(supplied_mass_cg))
            and np.allclose(supplied_mass_cg, expected, rtol=1e-8, atol=1e-6)
        )

    if not matches(False):
        if matches(True):
            raise ValueError(
                "Cutoff row retains the fairing; supply an explicit fairing-jettison state before TMI."
            )
        raise ValueError(
            "Cutoff mass/CG is incompatible with the detached, fairing-free upper-stage configuration."
        )
    initial_propellant = float(state[14])
    initial_ideal_delta_v = float(state[17])

    def energy(state):
        """Earth-relative C3 in km²/s²; negative values remain Earth-bound."""
        return (
            np.dot(state[3:6], state[3:6]) - 2 * MU / np.linalg.norm(state[:3])
        ) / 1e6

    def ground(time, state):
        return np.linalg.norm(state[:3]) - RE

    ground.terminal = True
    ground.direction = -1
    simulation.events = [{"event": "ascent_cutoff", "t_s": time}]
    status = "coast_complete"
    if coast_duration_s:
        state, time, solution = simulation.integrate(
            state, time, time + coast_duration_s, [ground]
        )
        if solution.t_events[0].size:
            status = "ground_contact"
    residual_propellant = run_config["vehicle"]["stages"][1]["residual_kg"]
    if status != "ground_contact":
        # An already-satisfied energy target requires no ignition or startup debit.
        if energy(state) >= target_c3_km2_s2:
            status = "C3_already_at_or_above_target"
        elif state[14] - restart_propellant_kg <= residual_propellant:
            status = "insufficient_restart_inventory"
        else:
            cg_before = simulation.vehicle.properties(state[13:15], False, False)[1]
            state[14] -= restart_propellant_kg
            cg_after = simulation.vehicle.properties(state[13:15], False, False)[1]
            cg_shift = matrix(state[6:10]) @ np.array([cg_after - cg_before, 0.0, 0.0])
            state[:3] += cg_shift
            state[3:6] += np.cross(matrix(state[6:10]) @ state[10:13], cg_shift)
            simulation.phase = 2
            simulation.cutoff = False
            simulation.ignition_time = time
            simulation.events.append(
                {
                    "event": "TMI_restart",
                    "t_s": time,
                    "startup_propellant_kg": restart_propellant_kg,
                }
            )

            def depleted(time, state):
                return state[14] - residual_propellant

            depleted.terminal = True
            depleted.direction = -1

            def target(time, state):
                return energy(state) - target_c3_km2_s2

            target.terminal = True
            target.direction = 1
            state, time, solution = simulation.integrate(
                state, time, time + max_burn_duration_s, [depleted, target, ground]
            )
            if solution.t_events[0].size:
                status = "residual_propellant_limit"
            elif solution.t_events[1].size:
                status = "target_C3_reached"
            elif solution.t_events[2].size:
                status = "ground_contact"
            else:
                status = "burn_time_limit"
    simulation.events.append({"event": status, "t_s": time})
    # integrate() retains the powered left-limit at the shutdown event for load
    # checks. Append a separate engine-off right-limit without changing the state.
    simulation.phase = 3
    simulation.cutoff = True
    simulation.rows.append(simulation.evaluate(time, state, True))
    history = pd.DataFrame(simulation.rows)
    final_c3 = float(energy(state))
    powered = history[history.thrust_N > 0]
    maximum_axial_g = float(history.axial_g.abs().max())
    maximum_lateral_g = float(history.lateral_g.abs().max())
    tracking_pass = None
    if not powered.empty:
        tracking_pass = bool(
            (powered.required_moment_Nm <= powered.available_moment_Nm + 1e-6).all()
        )
    return {
        "status": status,
        "target_C3_reached": bool(final_c3 >= target_c3_km2_s2 - 1e-7),
        "target_C3_km2_s2": float(target_c3_km2_s2),
        "achieved_C3_km2_s2": final_c3,
        "C3_margin_km2_s2": final_c3 - target_c3_km2_s2,
        "elapsed_from_ascent_cutoff_s": time - start_time,
        "remaining_propellant_kg": float(state[14]),
        "required_residual_kg": float(residual_propellant),
        "consumed_propellant_kg": initial_propellant - float(state[14]),
        "TMI_ideal_delta_v_m_s": float(state[17] - initial_ideal_delta_v),
        "events": simulation.events,
        "maximum_axial_g": maximum_axial_g,
        "maximum_lateral_g": maximum_lateral_g,
        "axial_acceleration_screen_pass": bool(
            maximum_axial_g <= run_config["limits"]["axial_g"]
        ),
        "lateral_acceleration_screen_pass": bool(
            maximum_lateral_g <= run_config["limits"]["lateral_g"]
        ),
        "powered_tracking_capacity_screen_pass": tracking_pass,
        "history": history.to_dict(orient="records"),
        "assumptions": [
            "Same shared 6DOF dynamics, atmosphere, moving mass, actuator and acceleration limits.",
            "Prograde departure guidance; no optimizer or planetary encounter matching.",
            "Supplied instantaneous restart propellant debit; no ignition shock or boiloff model.",
            "Source gimbal hardware and coast-control limitations remain applicable.",
        ],
    }


def integrated_tmi(
    *,
    coast_duration_s,
    restart_propellant_kg,
    target_c3_km2_s2,
    max_burn_duration_s,
    config=None,
    ascent_history=None,
    output_step_s=1.0,
    context=None,
):
    """Reuse or run the ascent, then continue from its actual orbital-speed cutoff."""
    if context is not None:
        ascent_history, _, simulation = context
        config = simulation.c
    run_config = copy.deepcopy(load_config() if config is None else config)
    if ascent_history is None:
        ascent, summary, simulation = run_trajectory(run_config)
        if not any(e["event"] == "orbital_speed_cutoff" for e in simulation.events):
            return {
                "status": "ascent_did_not_reach_cutoff",
                "target_C3_reached": False,
                "ascent_summary": summary,
            }
    else:
        ascent = pd.DataFrame(ascent_history)
        summary = None
    powered = ascent[(ascent.phase == 2) & (ascent.thrust_N > 0)]
    if powered.empty:
        raise ValueError("Ascent history contains no powered upper-stage state.")
    cutoff_state = powered.iloc[-1].to_dict()
    coast_after_cutoff = ascent[
        (ascent.t_s >= cutoff_state["t_s"]) & (ascent.phase == 3)
    ]
    if coast_after_cutoff.empty:
        raise ValueError(
            "Supplied ascent history does not demonstrate an orbital-cutoff coast."
        )
    result = finite_burn_tmi(
        cutoff_row=cutoff_state,
        config=run_config,
        coast_duration_s=coast_duration_s,
        restart_propellant_kg=restart_propellant_kg,
        target_c3_km2_s2=target_c3_km2_s2,
        max_burn_duration_s=max_burn_duration_s,
        output_step_s=output_step_s,
    )
    if summary is not None:
        result["ascent_summary"] = summary
    return result


def _stumpff(z):
    if abs(z) < 1e-6:
        return (
            0.5 - z / 24 + z * z / 720 - z**3 / 40320,
            1 / 6 - z / 120 + z * z / 5040 - z**3 / 362880,
        )
    x = math.sqrt(abs(z))
    if z > 0:
        return (1 - math.cos(x)) / z, (x - math.sin(x)) / x**3
    return (math.cosh(x) - 1) / (-z), (math.sinh(x) - x) / x**3


def lambert(
    *,
    departure_position_m,
    arrival_position_m,
    time_of_flight_s,
    central_mu_m3_s2,
    long_way=False,
):
    """Zero-revolution universal-variable solution; endpoints share one inertial frame.

    `long_way` chooses transfer angle > pi. Collinear endpoints are rejected;
    they require a specified transfer plane and a different singularity treatment.
    """
    departure_position = _vector(departure_position_m, "departure_position_m")
    arrival_position = _vector(arrival_position_m, "arrival_position_m")
    mu = _positive(central_mu_m3_s2, "central_mu_m3_s2")
    flight_time = _positive(time_of_flight_s, "time_of_flight_s")
    departure_radius = np.linalg.norm(departure_position)
    arrival_radius = np.linalg.norm(arrival_position)
    _positive(departure_radius, "departure radius")
    _positive(arrival_radius, "arrival radius")
    cos_angle = float(
        np.clip(
            np.dot(departure_position, arrival_position)
            / (departure_radius * arrival_radius),
            -1,
            1,
        )
    )
    sin_angle = math.sqrt(max(0.0, 1 - cos_angle * cos_angle)) * (-1 if long_way else 1)
    if abs(sin_angle) < 1e-9:
        raise ValueError(
            "Collinear Lambert endpoints require a specified transfer plane."
        )
    transfer_geometry = sin_angle * math.sqrt(
        departure_radius * arrival_radius / (1 - cos_angle)
    )

    def geometry(z):
        stumpff_c, stumpff_s = _stumpff(z)
        if stumpff_c <= 0:
            return None
        y = (
            departure_radius
            + arrival_radius
            + transfer_geometry * (z * stumpff_s - 1) / math.sqrt(stumpff_c)
        )
        if y < 0:
            return None
        elapsed = (
            (y / stumpff_c) ** 1.5 * stumpff_s + transfer_geometry * math.sqrt(y)
        ) / math.sqrt(mu)
        return elapsed, y

    # The zero-revolution branch ends at the first positive Stumpff singularity.
    grid = np.r_[
        -np.geomspace(1e4, 1e-8, 700),
        0.0,
        np.linspace(1e-8, 4 * math.pi**2 - 1e-6, 700),
    ]
    previous = None
    bracket = None
    for z in grid:
        value = geometry(float(z))
        if value is None:
            continue
        f = value[0] - flight_time
        if previous is not None and previous[1] * f <= 0:
            bracket = (previous[0], float(z))
            break
        previous = (float(z), f)
    if bracket is None:
        raise ValueError(
            "No zero-revolution Lambert root bracketed for these endpoints/time."
        )
    z = brentq(lambda z: geometry(z)[0] - flight_time, *bracket, xtol=1e-12, rtol=1e-13)
    elapsed, y = geometry(z)
    f = 1 - y / departure_radius
    g = transfer_geometry * math.sqrt(y / mu)
    gdot = 1 - y / arrival_radius
    departure_velocity = (arrival_position - f * departure_position) / g
    arrival_velocity = (gdot * arrival_position - departure_position) / g
    return {
        "departure_velocity_m_s": departure_velocity.tolist(),
        "arrival_velocity_m_s": arrival_velocity.tolist(),
        "time_of_flight_residual_s": float(elapsed - flight_time),
        "universal_parameter": float(z),
        "branch": (
            "zero-revolution long-way" if long_way else "zero-revolution short-way"
        ),
    }


def lambert_window(*, cases, central_mu_m3_s2):
    """Evaluate supplied ephemeris/date pairs; no fabricated planet ephemerides.

    Each case includes endpoint positions, planet velocities, flight time, and
    optional date labels. The caller must supply a common inertial frame/epoch.
    Declination is relative to the supplied XY plane; use equatorial ICRF states.
    """
    output = []
    if not cases:
        raise ValueError("At least one ephemeris case is required.")
    for case in cases:
        result = lambert(
            departure_position_m=case["departure_position_m"],
            arrival_position_m=case["arrival_position_m"],
            time_of_flight_s=case["time_of_flight_s"],
            central_mu_m3_s2=central_mu_m3_s2,
            long_way=case.get("long_way", False),
        )
        departure_excess = np.asarray(result["departure_velocity_m_s"]) - _vector(
            case["departure_planet_velocity_m_s"], "departure_planet_velocity_m_s"
        )
        arrival_excess = np.asarray(result["arrival_velocity_m_s"]) - _vector(
            case["arrival_planet_velocity_m_s"], "arrival_planet_velocity_m_s"
        )
        result.update(
            {
                "departure_label": case.get("departure_label", "supplied epoch"),
                "arrival_label": case.get("arrival_label", "supplied epoch"),
                "departure_C3_km2_s2": float(
                    np.dot(departure_excess, departure_excess) / 1e6
                ),
                "departure_excess_velocity_m_s": departure_excess.tolist(),
                "departure_declination_deg": float(
                    np.degrees(
                        np.arctan2(
                            departure_excess[2], np.linalg.norm(departure_excess[:2])
                        )
                    )
                ),
                "arrival_excess_speed_m_s": float(np.linalg.norm(arrival_excess)),
            }
        )
        output.append(result)
    return {
        "cases": output,
        "minimum_C3_case_index": int(
            np.argmin([x["departure_C3_km2_s2"] for x in output])
        ),
        "scope": "Minimum among supplied samples only; not a global launch-window optimization.",
    }
