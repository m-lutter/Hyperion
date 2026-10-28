"""Verification calculations consuming the current integrated six-DOF run."""

import numpy as np

from .trajectory import G0, run_trajectory


def flight_summary(context):
    """Extract requirement-facing measurements from one shared trajectory run.

    ``context`` is the tuple returned by ``current_flight()`` or
    ``run_trajectory()``. Result names carry units. Model assumptions and
    aerodynamic coverage are retained alongside the numerical measurements.
    """
    history, summary, simulation = context
    config = simulation.c
    powered = history[history.thrust_N > 0]
    result = {
        "payload_kg": config["vehicle"]["payload_kg"],
        "liftoff_mass_kg": float(history.mass_kg.iloc[0]),
        "installed_sea_level_thrust_N": config["vehicle"]["stages"][0]["thrust_N"],
        "liftoff_thrust_to_weight": float(
            history.thrust_N.iloc[0] / (history.mass_kg.iloc[0] * G0)
        ),
        "max_axial_g": float(history.axial_g.abs().max()),
        "max_lateral_g": float(history.lateral_g.abs().max()),
        "max_q_Pa": float(history.q_Pa.max()),
        "max_qalpha_Pa_deg": float(history.qalpha_Pa_deg.max()),
        "alpha_at_max_qalpha_deg": float(
            history.loc[history.qalpha_Pa_deg.idxmax(), "alpha_deg"]
        ),
        "minimum_control_ratio_margin_Nm": float(history.ratio2_margin_Nm.min()),
        "minimum_powered_control_ratio_margin_Nm": float(
            powered.ratio2_margin_Nm.min()
        ),
        "required_coast_control_Nm": summary["additional_coast_moment_for_ratio2_Nm"],
        "steering_loss_m_s": summary["cutoff_state"]["steering_loss_m_s"],
        "drag_loss_m_s": summary["cutoff_state"]["drag_loss_m_s"],
        "gravity_loss_m_s": summary["cutoff_state"]["gravity_loss_m_s"],
        "ideal_remaining_delta_v_m_s": summary["ideal_remaining_delta_v_m_s"],
        "length_m": sum(s["length_m"] for s in config["vehicle"]["stages"]),
        "diameter_m": config["vehicle"]["diameter_m"],
        "fineness_ratio": sum(s["length_m"] for s in config["vehicle"]["stages"])
        / config["vehicle"]["diameter_m"],
        "aero_outside_table_samples": summary["aero_outside_table_samples"],
        "aero_proxy_samples": summary["post_separation_aero_proxy_samples"],
        "events": simulation.events,
        "cutoff": summary["cutoff_state"],
        "assumptions": config["assumptions"],
    }
    for i, stage in enumerate(config["vehicle"]["stages"], 1):
        # Report actual integrated inventory rather than integrating discontinuities.
        result[f"stage{i}_propellant_used_kg"] = float(
            history[f"prop{i}_kg"].iloc[0] - history[f"prop{i}_kg"].iloc[-1]
        )
        result[f"stage{i}_dry_mass_kg"] = stage["dry_mass_kg"]
        result[f"stage{i}_thrust_N"] = stage["thrust_N"]
        result[f"stage{i}_isp_s"] = stage["isp_s"]
        result[f"stage{i}_loaded_propellant_kg"] = stage["propellant_kg"]
        result[f"stage{i}_propellant_fraction"] = stage["propellant_kg"] / (
            stage["propellant_kg"] + stage["dry_mass_kg"]
        )
    return result


def ascent_dispersions(context, cases, seed=20261028):
    """Rerun six-DOF cases; explicit overrides and seed, no invented dispersions.

    cases is a list of dictionaries of dot-path config changes (e.g.
    atmosphere.wind_reference_m_s). This is a finite scenario sweep; it is not
    a probability/confidence claim. Any Monte Carlo sample must include its
    user-supplied distribution and provenance outside this function.
    """
    import copy

    rows = []
    for case in cases:
        config = copy.deepcopy(context[2].c)
        for path, value in case.items():
            container = config
            keys = path.split(".")
            for key in keys[:-1]:
                container = (
                    container[int(key)]
                    if isinstance(container, list)
                    else container[key]
                )
            key = int(keys[-1]) if isinstance(container, list) else keys[-1]
            if isinstance(container, dict) and key not in container:
                raise ValueError(f"Unknown configuration path: {path}")
            container[key] = value
        config["solver"]["verbose"] = False
        history, summary, simulation = run_trajectory(config)
        rows.append(
            {
                "case": case,
                "cutoff": summary["cutoff_state"],
                "axial_g": float(history.axial_g.abs().max()),
                "lateral_g": float(history.lateral_g.max()),
                "q_Pa": float(history.q_Pa.max()),
                "qalpha_Pa_deg": float(history.qalpha_Pa_deg.max()),
                "minimum_control_margin_Nm": float(history.ratio2_margin_Nm.min()),
            }
        )
    return {
        "case_count": len(rows),
        "seed_record": seed,
        "cases": rows,
        "scope": "Explicit deterministic cases; not a validated stochastic population.",
    }


def sourced_sizing(context, tanks, component_mass_kg, allocated_mass_kg):
    """Recompute tank volumes/ullage and bottom-up dry mass against allocations.

    Each tank supplies name, volume_m3, loaded_mass_kg, density_kg_m3. Component
    totals must include fairing/adapter bookkeeping explicitly; do not compare an
    allocation to itself. Tank density and geometry are source design values.
    """
    from .propulsion import liquid_ullage

    rows = []
    for tank in tanks:
        # Validate the source values with the shared tank calculation.
        liquid_ullage(
            tank_volume_m3=tank["volume_m3"],
            loaded_liquid_mass_kg=tank["loaded_mass_kg"],
            liquid_density_kg_m3=tank["density_kg_m3"],
        )
        liquid = tank["loaded_mass_kg"] / tank["density_kg_m3"]
        rows.append(
            {
                "tank": tank["name"],
                "liquid_volume_m3": liquid,
                "tank_volume_m3": tank["volume_m3"],
                "ullage_fraction": 1 - liquid / tank["volume_m3"],
            }
        )
    components = np.asarray(component_mass_kg, dtype=float)
    allocations = np.asarray(allocated_mass_kg, dtype=float)
    if (
        components.shape != allocations.shape
        or not len(components)
        or not np.isfinite(components).all()
        or not np.isfinite(allocations).all()
        or np.any(components < 0)
        or np.any(allocations <= 0)
    ):
        raise ValueError(
            "Matching nonnegative component totals and positive allocations required."
        )
    return {
        "tanks": rows,
        "minimum_ullage_fraction": min(r["ullage_fraction"] for r in rows),
        "dry_mass_margin_kg": (allocations - components).tolist(),
        "minimum_dry_mass_margin_kg": float((allocations - components).min()),
        "basis": "Source bottom-up totals; fairing and replacement adapter hardware included explicitly. Not a measured mass record.",
    }


FUNCTIONS = {
    "flight_summary": flight_summary,
    "ascent_dispersions": ascent_dispersions,
    "sourced_sizing": sourced_sizing,
}
KERNEL_ACTIVITIES = {
    name: "flight_summary"
    for name in [
        "Six-DOF closed-loop flight simulation.",
        "Six-DOF closed-loop attitude simulation.",
        "Installed-thrust and liftoff-weight balance.",
        "Sea-level thrust-to-weight calculation.",
        "Rigid-body axial acceleration simulation.",
        "Control and aerodynamic moment balance.",
        "Vehicle length-to-diameter calculation.",
        "First-stage propellant-consumption integration.",
        "First-stage propellant-to-wet-mass calculation.",
        "Upper-stage propellant-to-wet-mass calculation.",
    ]
}
KERNEL_ACTIVITIES.update(
    {
        name: "ascent_dispersions"
        for name in [
            "Wind-dispersed six-DOF ascent simulation.",
            "Ascent dynamic-pressure and angle-of-attack sweep.",
            "Coupled first-stage throttle–trajectory simulation.",
            "Coupled upper-stage throttle–trajectory simulation.",
        ]
    }
)
KERNEL_ACTIVITIES.update(
    {
        name: "sourced_sizing"
        for name in [
            "Tank geometry and liquid-volume calculations.",
            "Ullage and loaded-liquid volume calculations.",
            "Stage dry-mass rollup.",
        ]
    }
)
