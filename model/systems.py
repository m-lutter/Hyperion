"""Reusable avionics, operations and programmatic engineering calculations.

Functions return named results and the assumptions needed to interpret them.
Start with ``rf_telemetry``, ``power_network``, or ``handling_analysis`` to combine
related checks; individual functions are also usable on their own. Units are in
parameter names and docstrings, including days/currency for schedule and cost.

All inputs are explicit. Results are calculations for an analyst to compare with
sourced criteria, not hardware qualification or launch authorization. Models do
not supply missing ICD limits, environmental rules, component ratings or FEA.
"""

from __future__ import annotations

import itertools
import math
import numpy as np
from scipy.linalg import expm
from scipy.integrate import trapezoid


def _array(value, name, ndim=None):
    a = np.asarray(value, dtype=float)
    if (
        a.size == 0
        or not np.all(np.isfinite(a))
        or (ndim is not None and a.ndim != ndim)
    ):
        raise ValueError(
            f"{name} must be a nonempty finite array"
            + (f" with {ndim} dimensions" if ndim is not None else "")
        )
    return a


def _positive(value, name, zero=False):
    a = _array(value, name)
    if np.any(a < 0 if zero else a <= 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return a


def _fraction(value, name):
    a = _positive(value, name, zero=True)
    if np.any(a > 1):
        raise ValueError(f"{name} must be between zero and one")
    return a


def _out(assumptions, **values):
    def convert(v):
        if isinstance(v, np.ndarray):
            return convert(v.tolist())
        if isinstance(v, np.generic):
            return convert(v.item())
        if isinstance(v, dict):
            return {k: convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("Model produced a nonfinite result")
        return v

    return convert(dict(assumptions=assumptions, **values))


def rf_link_budget(
    tx_positions_m,
    rx_position_m,
    frequency_hz,
    tx_power_w,
    tx_gain_dbi,
    rx_gain_dbi,
    loss_db,
    noise_temperature_k,
    bit_rate_bps,
    required_ebn0_db,
    minimum_elevation_deg,
):
    """Free-space RF link and ground antenna visibility; positions ECEF m, gains dBi, losses dB."""
    transmitter_positions = _array(tx_positions_m, "tx_positions_m", 2)
    receiver_position = _array(rx_position_m, "rx_position_m", 1)
    if transmitter_positions.shape[1] != 3 or receiver_position.shape != (3,):
        raise ValueError("ECEF positions require three components")
    frequency, power, temperature, rate = (
        float(_positive(v, n))
        for v, n in [
            (frequency_hz, "frequency_hz"),
            (tx_power_w, "tx_power_w"),
            (noise_temperature_k, "noise_temperature_k"),
            (bit_rate_bps, "bit_rate_bps"),
        ]
    )
    _positive(loss_db, "loss_db", zero=True)
    _array(
        [tx_gain_dbi, rx_gain_dbi, required_ebn0_db, minimum_elevation_deg],
        "link parameters",
    )
    if not 0 <= minimum_elevation_deg <= 90:
        raise ValueError("minimum_elevation_deg must lie in [0,90]")
    line_of_sight = transmitter_positions - receiver_position
    ranges = np.linalg.norm(line_of_sight, axis=1)
    if np.any(ranges <= 0) or np.linalg.norm(receiver_position) <= 0:
        raise ValueError("Nonzero range and station radius required")
    station_up = receiver_position / np.linalg.norm(receiver_position)
    sine_elevation = line_of_sight @ station_up / ranges
    elevation = np.rad2deg(np.arcsin(np.clip(sine_elevation, -1, 1)))
    free_space_loss = 20 * np.log10(4 * np.pi * ranges * frequency / 299792458.0)
    received_power = (
        10 * np.log10(power) + tx_gain_dbi + rx_gain_dbi - loss_db - free_space_loss
    )
    carrier_noise_density = received_power - 10 * np.log10(1.380649e-23 * temperature)
    energy_noise_ratio = carrier_noise_density - 10 * np.log10(rate)
    return _out(
        "Spherical-Earth ground horizon; specified constant gains/losses; no refraction, terrain, atmospheric absorption or antenna masking unless supplied in losses.",
        range_m=ranges,
        elevation_deg=elevation,
        visible=elevation >= minimum_elevation_deg,
        received_dbw=received_power,
        cn0_dbhz=carrier_noise_density,
        ebn0_db=energy_noise_ratio,
        link_margin_db=energy_noise_ratio - required_ebn0_db,
    )


def satellite_visibility(
    receiver_positions_m, satellite_positions_m, minimum_elevation_deg, earth_radius_m
):
    """GNSS geometry at simultaneous epochs; ECEF m arrays [T,3] and [T,S,3], mask degrees."""
    receiver_positions, satellite_positions = _array(
        receiver_positions_m, "receiver_positions_m", 2
    ), _array(satellite_positions_m, "satellite_positions_m", 3)
    radius = float(_positive(earth_radius_m, "earth_radius_m"))
    if (
        receiver_positions.shape[1] != 3
        or satellite_positions.shape[0] != receiver_positions.shape[0]
        or satellite_positions.shape[2] != 3
    ):
        raise ValueError("Positions require [T,3] and [T,S,3]")
    _array(minimum_elevation_deg, "minimum_elevation_deg")
    if not -90 <= minimum_elevation_deg <= 90 or np.any(
        np.linalg.norm(receiver_positions, axis=1) < radius
    ):
        raise ValueError("Invalid elevation mask or receiver below spherical surface")
    line_of_sight = satellite_positions - receiver_positions[:, None, :]
    distance = np.linalg.norm(line_of_sight, axis=2)
    if np.any(distance <= 0):
        raise ValueError("Satellite cannot coincide with receiver")
    directions = line_of_sight / distance[:, :, None]
    local_up = receiver_positions / np.linalg.norm(receiver_positions, axis=1)[:, None]
    sine_elevation = np.einsum("tij,tj->ti", directions, local_up)
    elevation = np.rad2deg(np.arcsin(np.clip(sine_elevation, -1, 1)))

    # Project Earth's center onto each receiver-to-satellite line segment.
    # A segment that passes through Earth is hidden even above the angle mask.
    receiver_grid = receiver_positions[:, None, :] + np.zeros_like(line_of_sight)
    projection = -np.einsum("tij,tij->ti", receiver_grid, line_of_sight) / distance**2
    segment_fraction = np.clip(projection, 0, 1)
    closest = (
        receiver_positions[:, None, :] + segment_fraction[:, :, None] * line_of_sight
    )
    visible = (elevation >= minimum_elevation_deg) & (
        np.linalg.norm(closest, axis=2) >= radius - 1e-6
    )
    gdop = []
    for epoch_directions, mask in zip(directions, visible):
        geometry_matrix = np.column_stack(
            (-epoch_directions[mask], np.ones(mask.sum()))
        )
        has_full_geometry = (
            len(geometry_matrix) >= 4 and np.linalg.matrix_rank(geometry_matrix) == 4
        )
        if has_full_geometry:
            covariance = np.linalg.inv(geometry_matrix.T @ geometry_matrix)
            gdop.append(float(np.sqrt(np.trace(covariance))))
        else:
            gdop.append(None)
    return _out(
        "Known simultaneous satellite ephemerides; spherical occultation and elevation mask; no signal-strength or receiver acquisition model.",
        visible=visible,
        satellite_count=visible.sum(axis=1),
        elevation_deg=elevation,
        gdop=gdop,
    )


def telemetry_budget(
    streams,
    link_rate_bps,
    coding_rate,
    protocol_efficiency,
    propagation_delay_s,
    processing_delay_s,
):
    """Packet budget; streams contain packet_bits and packets_per_s; times s, link rate bits/s."""
    rate = float(_positive(link_rate_bps, "link_rate_bps"))
    coding = float(_fraction(coding_rate, "coding_rate"))
    efficiency = float(_fraction(protocol_efficiency, "protocol_efficiency"))
    if coding * efficiency == 0 or not streams:
        raise ValueError("Positive coding/efficiency and at least one stream required")
    bits = np.array(
        [float(_positive(x["packet_bits"], "packet_bits")) for x in streams]
    )
    hz = np.array(
        [
            float(_positive(x["packets_per_s"], "packets_per_s", zero=True))
            for x in streams
        ]
    )
    delays = _positive([propagation_delay_s, processing_delay_s], "delays", zero=True)
    useful = rate * coding * efficiency
    demand = float(bits @ hz)
    return _out(
        "Steady packet rates; no contention or queueing model. Latency is serialization plus supplied propagation/processing, a lower bound under load.",
        demanded_bps=demand,
        useful_capacity_bps=useful,
        utilization=demand / useful,
        spare_capacity_bps=useful - demand,
        packet_latency_lower_bound_s=bits / useful + delays.sum(),
    )


def power_energy(
    loads,
    capacity_wh,
    usable_fraction,
    distribution_efficiency,
    reserve_fraction,
    transfer_time_s,
    allowed_transfer_time_s,
):
    """Electrical energy budget; loads contain power_w, duration_s, duty_fraction; capacity Wh, transfer times s."""
    cap = float(_positive(capacity_wh, "capacity_wh"))
    usable, eff, reserve = (
        float(_fraction(v, n))
        for v, n in [
            (usable_fraction, "usable_fraction"),
            (distribution_efficiency, "distribution_efficiency"),
            (reserve_fraction, "reserve_fraction"),
        ]
    )
    if eff == 0 or not loads:
        raise ValueError("Nonzero efficiency and loads required")
    powers, energy = [], []
    for row in loads:
        p = float(_positive(row["power_w"], "power_w", zero=True))
        t = float(_positive(row["duration_s"], "duration_s", zero=True))
        duty = float(_fraction(row["duty_fraction"], "duty_fraction"))
        powers.append(p)
        energy.append(p * t * duty / 3600 / eff)
    _positive([transfer_time_s, allowed_transfer_time_s], "transfer times", zero=True)
    available = cap * usable * (1 - reserve)
    return _out(
        "Constant load segments and specified efficiencies; sum of powers is conservative simultaneous peak; temperature/aging belong in usable_fraction.",
        required_wh=sum(energy),
        available_wh=available,
        energy_margin_wh=available - sum(energy),
        per_load_wh=energy,
        simultaneous_peak_w=sum(powers) / eff,
        transfer_margin_s=allowed_transfer_time_s - transfer_time_s,
    )


def dc_circuit(
    source_voltage_v,
    source_resistance_ohm,
    conductor_resistance_ohm_m,
    one_way_length_m,
    load_current_a,
    ampacity_a,
    fault_path_resistance_ohm,
    protection_trip_current_a,
    protection_clearing_s,
    allowed_clearing_s,
):
    """Resistive two-wire DC voltage drop, heating, bolted fault and supplied clearing time; SI units."""
    voltage = float(_positive(source_voltage_v, "source_voltage_v"))
    source, wire, length, current, fault = _positive(
        [
            source_resistance_ohm,
            conductor_resistance_ohm_m,
            one_way_length_m,
            load_current_a,
            fault_path_resistance_ohm,
        ],
        "circuit inputs",
        zero=True,
    )
    amp, trip = _positive([ampacity_a, protection_trip_current_a], "ratings")
    _positive([protection_clearing_s, allowed_clearing_s], "clearing times", zero=True)
    total = source + 2 * wire * length
    if total + fault <= 0:
        raise ValueError("Fault path resistance must be positive")
    drop = current * total
    fault_i = voltage / (total + fault)
    return _out(
        "Lumped DC resistance; supplied current and protection clearing time, no arc, inrush, AC impedance or battery dynamics.",
        voltage_drop_v=drop,
        load_voltage_v=voltage - drop,
        wire_heating_w=current**2 * 2 * wire * length,
        ampacity_margin_a=amp - current,
        fault_current_a=fault_i,
        trip_current_margin_a=fault_i - trip,
        clearing_margin_s=allowed_clearing_s - protection_clearing_s,
    )


def timing_budget(stages_s, deadline_s, jitter_s):
    """Worst-case serial event/umbilical/command timing; stage durations, deadline and additive jitter in s."""
    durations = _positive(stages_s, "stages_s", zero=True)
    _positive([deadline_s, jitter_s], "timing inputs", zero=True)
    total = float(durations.sum() + jitter_s)
    return _out(
        "Serial critical chain; stages are supplied worst-case bounds, additive jitter counted once.",
        completion_s=total,
        deadline_s=deadline_s,
        margin_s=deadline_s - total,
        stage_completion_s=np.cumsum(durations),
    )


def state_machine(initial_state, events, transitions, forbidden_states):
    """Evaluate explicit transition/guard logic; events {time_s,event,signals}, transitions {from,event,to,guards}."""
    state, history, rejected = str(initial_state), [], []
    last_time = -math.inf
    for event in events:
        time = float(_positive(event["time_s"], "time_s", zero=True))
        if time < last_time:
            raise ValueError("Events must be time ordered")
        last_time = time
        candidates = [
            tr
            for tr in transitions
            if tr["from"] == state and tr["event"] == event["event"]
        ]
        eligible = [
            tr
            for tr in candidates
            if all(
                k in event["signals"] and event["signals"][k] == v
                for k, v in tr["guards"].items()
            )
        ]
        if len(eligible) > 1:
            raise ValueError("Ambiguous eligible transitions")
        previous = state
        if eligible:
            state = eligible[0]["to"]
        else:
            rejected.append(
                dict(
                    time_s=time,
                    state=state,
                    event=event["event"],
                    reason="No transition with satisfied, present guards",
                )
            )
        history.append(
            dict(time_s=time, previous=previous, state=state, event=event["event"])
        )
    violations = [h for h in history if h["state"] in forbidden_states]
    if initial_state in forbidden_states:
        violations.insert(0, dict(time_s=0, state=initial_state, event="initial"))
    return _out(
        "Only supplied deterministic transition table and event cases are exercised; no completeness or physical reliability claim.",
        final_state=state,
        history=history,
        rejected_events=rejected,
        forbidden_state_visits=violations,
    )


def launch_commit_rules(measurements, rules):
    """Evaluate supplied launch/environment predicates; rules {name,measurement,operator,limit}; units must match."""
    operators = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    if not rules:
        raise ValueError("At least one explicit rule required")
    results = []
    for rule in rules:
        if rule["operator"] not in operators:
            raise ValueError("Unsupported operator")
        name = rule["measurement"]
        if name not in measurements or measurements[name] is None:
            result = None
        else:
            _array(measurements[name], name)
            _array(rule["limit"], "limit")
            result = bool(
                operators[rule["operator"]](measurements[name], rule["limit"])
            )
        results.append(dict(name=rule["name"], satisfied=result))
    # Unknown is never treated as permission. Actual authorization is outside this model.
    return _out(
        "Analyst-supplied predicates; no embedded FAA/NASA criteria, sensor validity or launch authority.",
        predicates=results,
        modeled_all_satisfied=all(x["satisfied"] is True for x in results),
        missing_measurements=[
            r["measurement"] for r, x in zip(rules, results) if x["satisfied"] is None
        ],
    )


def navigation_covariance(
    transition_matrix,
    process_covariance,
    measurement_matrix,
    measurement_covariance,
    initial_covariance,
    initial_bias,
    steps,
    samples,
    seed,
):
    """Discrete linear Kalman covariance/bias propagation and final Gaussian Monte Carlo; use consistent state units."""
    transition, process_noise, measurement, measurement_noise, covariance = [
        _array(x, state_count, 2)
        for x, state_count in [
            (transition_matrix, "F"),
            (process_covariance, "Q"),
            (measurement_matrix, "H"),
            (measurement_covariance, "R"),
            (initial_covariance, "P0"),
        ]
    ]
    bias = _array(initial_bias, "initial_bias", 1)
    state_count = len(bias)
    if (
        transition.shape != (state_count, state_count)
        or process_noise.shape != (state_count, state_count)
        or covariance.shape != (state_count, state_count)
        or measurement.shape[1] != state_count
        or measurement_noise.shape != (measurement.shape[0], measurement.shape[0])
    ):
        raise ValueError("Incompatible state/measurement dimensions")
    if (
        not isinstance(steps, int)
        or steps < 1
        or not isinstance(samples, int)
        or samples < 2
    ):
        raise ValueError("steps>=1 and samples>=2 must be integers")
    for matrix, name in [
        (process_noise, "Q"),
        (measurement_noise, "R"),
        (covariance, "P0"),
    ]:
        if (
            not np.allclose(matrix, matrix.T)
            or np.linalg.eigvalsh(matrix).min() < -1e-10
        ):
            raise ValueError(f"{name} must be symmetric positive semidefinite")
    history = []
    for _ in range(steps):
        prior = transition @ covariance @ transition.T + process_noise
        innovation = measurement @ prior @ measurement.T + measurement_noise
        if np.linalg.eigvalsh(innovation).min() <= 0:
            raise ValueError("Innovation covariance must be positive definite")
        gain = np.linalg.solve(innovation, measurement @ prior).T
        correction = np.eye(state_count) - gain @ measurement
        covariance = (
            correction @ prior @ correction.T + gain @ measurement_noise @ gain.T
        )  # Joseph update retains numerical PSD.
        covariance = (covariance + covariance.T) / 2
        bias = correction @ transition @ bias
        history.append(np.sqrt(np.maximum(0, np.diag(covariance))))
    draws = np.random.default_rng(seed).multivariate_normal(
        bias, covariance, size=samples
    )
    return _out(
        "Fixed discrete linear model, independent Gaussian white noises and measurement at every step; final-state Monte Carlo samples this covariance, not nonlinear mission dynamics.",
        covariance=covariance,
        bias=bias,
        sigma_history=history,
        sample_mean=draws.mean(axis=0),
        sample_covariance=np.atleast_2d(np.cov(draws, rowvar=False)),
        absolute_error_95=np.quantile(np.abs(draws), 0.95, axis=0),
    )


def frequency_margins(frequency_hz, open_loop_real, open_loop_imag):
    """Sampled SISO open-loop gain/phase crossover margins; frequencies Hz, complex loop gain dimensionless."""
    f, re, im = (
        _array(v, n, 1)
        for v, n in [
            (frequency_hz, "frequency_hz"),
            (open_loop_real, "real"),
            (open_loop_imag, "imag"),
        ]
    )
    if (
        f.shape != re.shape
        or f.shape != im.shape
        or len(f) < 2
        or np.any(f <= 0)
        or np.any(np.diff(f) <= 0)
    ):
        raise ValueError(
            "Same-length arrays with positive increasing frequencies required"
        )
    mag = np.hypot(re, im)
    if np.any(mag <= 0):
        raise ValueError("Nonzero sampled loop gain required")
    db = 20 * np.log10(mag)
    phase = np.rad2deg(np.unwrap(np.angle(re + 1j * im)))
    logf = np.log(f)

    def crossings(values, target):
        out = []
        for i in range(len(values) - 1):
            if values[i] == target:
                out.append((i, 0.0))
            elif (values[i] - target) * (values[i + 1] - target) < 0:
                out.append(
                    (i, float((target - values[i]) / (values[i + 1] - values[i])))
                )
        if values[-1] == target:
            out.append((len(values) - 2, 1.0))
        return out

    gains = []
    phases = []
    for i, t in crossings(db, 0):
        ph = (1 - t) * phase[i] + t * phase[i + 1]
        phases.append(
            dict(
                frequency_hz=float(np.exp((1 - t) * logf[i] + t * logf[i + 1])),
                phase_margin_deg=180 + ph,
            )
        )
    low = int(math.floor((phase.min() + 180) / 360)) - 1
    high = int(math.ceil((phase.max() + 180) / 360)) + 1
    for k in range(low, high + 1):
        for i, t in crossings(phase, -180 + 360 * k):
            gains.append(
                dict(
                    frequency_hz=float(np.exp((1 - t) * logf[i] + t * logf[i + 1])),
                    gain_margin_db=-float((1 - t) * db[i] + t * db[i + 1]),
                )
            )
    return _out(
        "Interpolated sampled SISO loop data; all odd-180-degree crossings retained. No crossings means unknown outside the sampled band, not infinite stability. Open-loop unstable poles and Nyquist encirclements need separate assessment.",
        gain_crossovers=phases,
        phase_crossovers=gains,
        minimum_phase_margin_deg=min(
            (x["phase_margin_deg"] for x in phases), default=None
        ),
        minimum_gain_margin_db=min((x["gain_margin_db"] for x in gains), default=None),
    )


def linear_feedback(
    a_matrix, b_matrix, feedback_gain, initial_state, duration_s, sample_interval_s
):
    """Continuous linear state-feedback free response xdot=(A-BK)x; duration/sample interval s, caller state units."""
    plant, input_matrix, gain = (
        _array(v, n, 2)
        for v, n in [(a_matrix, "A"), (b_matrix, "B"), (feedback_gain, "K")]
    )
    initial = _array(initial_state, "x0", 1)
    if (
        plant.shape != (len(initial), len(initial))
        or input_matrix.shape[0] != len(initial)
        or gain.shape != (input_matrix.shape[1], len(initial))
    ):
        raise ValueError("Incompatible state/input dimensions")
    duration, sample_step = _positive([duration_s, sample_interval_s], "times")
    if duration / sample_step > 100000:
        raise ValueError("At most 100000 sample intervals allowed")
    times = np.arange(0, duration, sample_step).tolist() + [float(duration)]
    closed_loop = plant - input_matrix @ gain
    eigenvalues = np.linalg.eigvals(closed_loop)
    states = np.array([expm(closed_loop * t) @ initial for t in times])
    return _out(
        "Unsaturated time-invariant linear plant and full-state feedback; no nonlinear dynamics, estimator delay or actuator limits.",
        time_s=times,
        state=states,
        command=-states @ gain.T,
        eigenvalues_real=eigenvalues.real,
        eigenvalues_imag=eigenvalues.imag,
        asymptotically_stable=bool(np.max(eigenvalues.real) < 0),
    )


def fault_tree(event_probabilities, tree):
    """Exact Boolean fault-tree probability under independent basic events; tree has AND/OR/NOT children or event name."""
    names = list(event_probabilities)
    if not names or len(names) > 18:
        raise ValueError("Supply 1 to 18 independent basic events")
    probabilities = {k: float(_fraction(v, k)) for k, v in event_probabilities.items()}

    def evaluate(node, assignment):
        if isinstance(node, str):
            if node not in assignment:
                raise ValueError(f"Unknown basic event {node}")
            return assignment[node]
        gate = node["gate"]
        children = node["children"]
        if not children or (gate == "NOT" and len(children) != 1):
            raise ValueError("Invalid gate children")
        values = [evaluate(c, assignment) for c in children]
        if gate == "AND":
            return all(values)
        if gate == "OR":
            return any(values)
        if gate == "NOT":
            return not values[0]
        raise ValueError("Gate must be AND, OR, or NOT")

    total = 0.0
    for bits in itertools.product([False, True], repeat=len(names)):
        if evaluate(tree, dict(zip(names, bits))):
            total += math.prod(
                probabilities[k] if bit else 1 - probabilities[k]
                for k, bit in zip(names, bits)
            )
    return _out(
        "Independent basic-event probabilities over one explicitly chosen exposure interval. Repeated events share a single Boolean variable; common cause must be an explicit event.",
        top_event_probability=min(1.0, max(0.0, total)),
        basic_event_count=len(names),
    )


def fmea_ranking(failures, event_probabilities, tree):
    """Analyst FMEA risk-priority rankings (severity*occurrence*detection) plus independent-event fault tree."""
    if not failures:
        raise ValueError("At least one failure required")
    ranked = []
    for f in failures:
        scores = _positive(
            [f["severity"], f["occurrence"], f["detection"]], "FMEA scores"
        )
        ranked.append(dict(id=f["id"], risk_priority_number=float(np.prod(scores))))
    result = fault_tree(event_probabilities, tree)
    return _out(
        "FMEA scores are supplied ordinal rankings, not failure probabilities; no universal acceptance threshold. Fault tree uses independent basic-event probabilities.",
        ranked_failures=sorted(
            ranked, key=lambda r: r["risk_priority_number"], reverse=True
        ),
        top_event_probability=result["top_event_probability"],
    )


def critical_path(tasks):
    """Precedence-network early/late dates and float; tasks {id,duration_days,predecessors}, durations days."""
    if not tasks:
        raise ValueError("Tasks required")
    records = {t["id"]: t for t in tasks}
    if len(records) != len(tasks):
        raise ValueError("Duplicate task ID")
    for t in tasks:
        _positive(t["duration_days"], "duration_days", zero=True)
        if any(p not in records for p in t["predecessors"]):
            raise ValueError("Unknown predecessor")
    order = []
    earliest_start = {}
    earliest_finish = {}
    while len(order) < len(tasks):
        ready = [
            key
            for key, t in records.items()
            if key not in earliest_finish
            and all(p in earliest_finish for p in t["predecessors"])
        ]
        if not ready:
            raise ValueError("Task network contains a cycle")
        for key in ready:
            earliest_start[key] = max(
                (earliest_finish[p] for p in records[key]["predecessors"]), default=0.0
            )
            earliest_finish[key] = earliest_start[key] + records[key]["duration_days"]
            order.append(key)
    project_finish = max(earliest_finish.values())
    latest_start = {}
    for key in reversed(order):
        successors = [s for s in order if key in records[s]["predecessors"]]
        latest_start[key] = (
            min((latest_start[s] for s in successors), default=project_finish)
            - records[key]["duration_days"]
        )
    return _out(
        "Deterministic precedence-only schedule; no resource leveling, calendars, task uncertainty or schedule risk simulation.",
        duration_days=project_finish,
        tasks=[
            dict(
                id=k,
                early_start_days=earliest_start[k],
                early_finish_days=earliest_finish[k],
                late_start_days=latest_start[k],
                float_days=latest_start[k] - earliest_start[k],
            )
            for k in order
        ],
        critical_tasks=[
            k for k in order if abs(latest_start[k] - earliest_start[k]) < 1e-9
        ],
    )


def parametric_cost(
    development_cost,
    first_unit_cost,
    learning_curve_fraction,
    quantity,
    per_launch_operations_cost,
    overhead_fraction,
    price_margin_fraction,
):
    """Unit learning-curve recurring/development cost; common currency, quantity integer, learning slope at each production doubling."""
    development, first_unit, operations = _positive(
        [development_cost, first_unit_cost, per_launch_operations_cost],
        "costs",
        zero=True,
    )
    slope = float(_fraction(learning_curve_fraction, "learning_curve_fraction"))
    overhead = float(_positive(overhead_fraction, "overhead_fraction", zero=True))
    margin = float(_fraction(price_margin_fraction, "price_margin_fraction"))
    if slope == 0 or margin >= 1 or not isinstance(quantity, int) or quantity < 1:
        raise ValueError("Invalid slope, gross margin, or quantity")
    units = first_unit * np.arange(1, quantity + 1) ** (math.log2(slope))
    recurring = (units + operations) * (1 + overhead)
    total = development + recurring.sum()
    return _out(
        "Unit-cost power law, constant operations/overhead and no inflation/discounting; supplied estimates need sourced basis. Price uses gross revenue margin.",
        unit_recurring_cost=recurring,
        total_program_cost=total,
        average_fully_loaded_cost=total / quantity,
        average_launch_price=total / quantity / (1 - margin),
    )


def transport_clearance(
    vehicle_envelope_m,
    route_envelopes_m,
    payload_mass_kg,
    capacity_kg,
    clearance_allowance_m,
):
    """Axis-aligned route envelope and transporter capacity; envelopes [width,height,length] m, masses kg."""
    vehicle = _positive(vehicle_envelope_m, "vehicle_envelope_m")
    route = _positive(route_envelopes_m, "route_envelopes_m")
    if vehicle.shape != (3,) or route.ndim != 2 or route.shape[1] != 3:
        raise ValueError("Envelope shapes [3] and [N,3] required")
    _positive(
        [payload_mass_kg, capacity_kg, clearance_allowance_m],
        "transport inputs",
        zero=True,
    )
    margin = route - vehicle - 2 * clearance_allowance_m
    return _out(
        "Straight-route axis-aligned envelope screen; clearance allowance applied per side. Turning swept paths, bridges, axle loads and restraint dynamics require separate models.",
        clearance_margin_m=margin,
        minimum_clearance_margin_m=float(margin.min()),
        capacity_margin_kg=capacity_kg - payload_mass_kg,
    )


def lift_reactions(
    mass_kg,
    cg_position_m,
    support_positions_m,
    sling_angles_deg,
    dynamic_factor,
    sling_ratings_n,
):
    """Two-point static lift/transport support reactions and sling tension; CG/support stations m, angles from vertical degrees, ratings N."""
    mass = float(_positive(mass_kg, "mass_kg"))
    dynamic = float(_positive(dynamic_factor, "dynamic_factor"))
    support = _array(support_positions_m, "support_positions_m", 1)
    angles = _array(sling_angles_deg, "sling_angles_deg", 1)
    ratings = _positive(sling_ratings_n, "sling_ratings_n")
    _array(cg_position_m, "cg_position_m")
    if (
        support.shape != (2,)
        or angles.shape != (2,)
        or ratings.shape != (2,)
        or support[1] <= support[0]
        or np.any(np.abs(angles) >= 90)
    ):
        raise ValueError(
            "Two ordered supports and angles within +/-90 degrees required"
        )
    if not support[0] <= cg_position_m <= support[1]:
        raise ValueError("CG outside supports requires a restraint/tipping model")
    weight = mass * 9.80665 * dynamic
    reactions = (
        weight
        * np.array([support[1] - cg_position_m, cg_position_m - support[0]])
        / (support[1] - support[0])
    )
    tension = reactions / np.cos(np.deg2rad(angles))
    return _out(
        "Rigid object, two coplanar support points and externally supplied dynamic factor; sling angles measured from vertical.",
        support_reaction_n=reactions,
        sling_tension_n=tension,
        sling_margin_n=ratings - tension,
    )


def tolerance_stack(
    nominal_clearance_m,
    signed_contributions_m,
    absolute_tolerances_m,
    required_clearance_m,
):
    """Worst-case interval clearance/tolerance stack; all lengths m. RSS also reported for independent centered tolerance terms."""
    terms = _array(signed_contributions_m, "signed_contributions_m", 1)
    tol = _positive(absolute_tolerances_m, "absolute_tolerances_m", zero=True)
    if terms.shape != tol.shape:
        raise ValueError("Contributions/tolerances shape mismatch")
    _array([nominal_clearance_m, required_clearance_m], "clearances")
    nominal = nominal_clearance_m + terms.sum()
    worst = nominal - tol.sum()
    return _out(
        "Worst-case algebraic interval stack; RSS is a comparison statistic only, not a guaranteed tolerance or confidence limit.",
        nominal_clearance_m=nominal,
        minimum_clearance_m=worst,
        worst_case_margin_m=worst - required_clearance_m,
        rss_tolerance_m=np.linalg.norm(tol),
    )


def concentration_balance(
    volume_m3,
    ventilation_m3_s,
    inlet_concentration,
    initial_concentration,
    source_per_s,
    duration_s,
    concentration_limit,
):
    """Perfectly mixed vapor/particle balance; concentrations amount/m^3, source amount/s, constant ventilation m^3/s."""
    volume = float(_positive(volume_m3, "volume_m3"))
    flow, source, time = _positive(
        [ventilation_m3_s, source_per_s, duration_s], "flow/source/time", zero=True
    )
    inlet, initial, limit = _positive(
        [inlet_concentration, initial_concentration, concentration_limit],
        "concentrations",
        zero=True,
    )
    final = (
        inlet
        + source / flow
        + (initial - inlet - source / flow) * np.exp(-flow * time / volume)
        if flow > 0
        else initial + source * time / volume
    )
    peak = max(initial, final)
    return _out(
        "Uniform perfectly mixed volume; constant source/flow, no deposition, stratification, chemical reaction or local plume concentration.",
        final_concentration=final,
        peak_concentration=peak,
        concentration_margin=limit - peak,
        exchanged_volume_m3=flow * time,
    )


def humidity_balance(
    dry_air_mass_kg,
    dry_air_flow_kg_s,
    inlet_humidity_ratio,
    initial_humidity_ratio,
    moisture_source_kg_s,
    duration_s,
    pressure_pa,
    temperature_k,
):
    """Mixed dry-air water mass balance; humidity ratio kg water/kg dry air; fixed air mass, pressure Pa and temperature K."""
    mass = float(_positive(dry_air_mass_kg, "dry_air_mass_kg"))
    flow, source, time = _positive(
        [dry_air_flow_kg_s, moisture_source_kg_s, duration_s],
        "flow/source/time",
        zero=True,
    )
    win, w0 = _positive(
        [inlet_humidity_ratio, initial_humidity_ratio], "humidity ratios", zero=True
    )
    pressure = float(_positive(pressure_pa, "pressure_pa"))
    temp = float(_positive(temperature_k, "temperature_k"))
    if not 273.15 <= temp <= 323.15:
        raise ValueError("Saturation approximation valid here only for 0 to 50 degC")
    w = (
        win + source / flow + (w0 - win - source / flow) * np.exp(-flow * time / mass)
        if flow > 0
        else w0 + source * time / mass
    )
    vapor = pressure * w / (0.62198 + w)
    tc = temp - 273.15
    saturation = 610.94 * np.exp(17.625 * tc / (tc + 243.04))
    return _out(
        "Well-mixed dry-air mass, fixed temperature/pressure, no condensation removal. RH above 100% flags model breakdown/condensation demand; approximate Magnus saturation over 0-50 degC.",
        final_humidity_ratio=w,
        vapor_partial_pressure_pa=vapor,
        relative_humidity_percent=100 * vapor / saturation,
        condensation_indicated=bool(vapor > saturation),
    )


def thermal_response(
    thermal_capacity_j_k,
    conductance_w_k,
    initial_temperature_k,
    ambient_temperature_k,
    heat_input_w,
    duration_s,
):
    """Single-node transient heat balance C*dT/dt=Q+UA*(Tamb-T); C J/K, UA W/K, temperatures K, duration s."""
    c = float(_positive(thermal_capacity_j_k, "thermal_capacity_j_k"))
    ua, time = _positive([conductance_w_k, duration_s], "conductance/time", zero=True)
    t0, ambient = _positive(
        [initial_temperature_k, ambient_temperature_k], "temperatures"
    )
    q = float(_array(heat_input_w, "heat_input_w"))
    final = (
        ambient + q / ua + (t0 - ambient - q / ua) * np.exp(-ua * time / c)
        if ua > 0
        else t0 + q * time / c
    )
    if final <= 0:
        raise ValueError("Model predicts nonphysical absolute temperature")
    return _out(
        "Lumped isothermal body with constant conductance/heat input; no radiation nonlinearity, phase change or thermal gradients.",
        final_temperature_k=final,
        maximum_temperature_k=max(t0, final),
        minimum_temperature_k=min(t0, final),
        stored_energy_change_j=c * (final - t0),
    )


def weather_separation(
    event_positions_m, flight_path_m, event_ages_s, exclusion_distance_m, waiting_time_s
):
    """Lightning/storm event distance to polyline and elapsed-time screen; common Cartesian frame m and ages s."""
    events = _array(event_positions_m, "event_positions_m", 2)
    path = _array(flight_path_m, "flight_path_m", 2)
    ages = _positive(event_ages_s, "event_ages_s", zero=True)
    _positive([exclusion_distance_m, waiting_time_s], "rule thresholds", zero=True)
    if (
        events.shape[1] != 3
        or path.shape[1] != 3
        or len(path) < 2
        or ages.shape != (len(events),)
    ):
        raise ValueError("Require [N,3] events, [M>=2,3] path, N ages")
    distances = []
    for event in events:
        candidates = []
        for a, b in zip(path[:-1], path[1:]):
            d = b - a
            den = float(d @ d)
            closest = a + np.clip((event - a) @ d / den, 0, 1) * d if den else a
            candidates.append(np.linalg.norm(event - closest))
        distances.append(min(candidates))
    distances = np.asarray(distances)
    return _out(
        "Euclidean point-to-polyline distance in supplied common frame; screening rule is distance<=exclusion AND age<waiting time. No prescribed lightning/weather policy supplied.",
        minimum_distance_m=distances,
        exclusion_active=(distances <= exclusion_distance_m) & (ages < waiting_time_s),
    )


def pressure_trend(time_s, pressure_pa, forecast_horizon_s):
    """Least-squares pressure trend and linear projection; time s, pressure Pa, future horizon s."""
    t, p = _array(time_s, "time_s", 1), _positive(pressure_pa, "pressure_pa")
    if t.shape != p.shape or len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError("At least three increasing time samples required")
    horizon = float(_positive(forecast_horizon_s, "forecast_horizon_s", zero=True))
    x = t - t.mean()
    slope = float(x @ (p - p.mean()) / (x @ x))
    fitted = p.mean() + slope * x
    residual = p - fitted
    return _out(
        "Linear regression/extrapolation only; no meteorological forecast skill or causal weather classification implied.",
        slope_pa_s=slope,
        projected_pressure_pa=float(p.mean() + slope * (t[-1] + horizon - t.mean())),
        residual_rms_pa=float(np.sqrt(np.mean(residual**2))),
    )


def actuator_response(
    time_s,
    command,
    initial_position,
    time_constant_s,
    rate_limit_per_s,
    lower_limit,
    upper_limit,
):
    """First-order, rate/position-limited actuator with zero-order-held commands; times s, positions/commands in common units."""
    t, u = _array(time_s, "time_s", 1), _array(command, "command", 1)
    tau, rate = _positive([time_constant_s, rate_limit_per_s], "time constant/rate")
    _array([initial_position, lower_limit, upper_limit], "position limits")
    if (
        t.shape != u.shape
        or len(t) < 2
        or np.any(np.diff(t) <= 0)
        or lower_limit >= upper_limit
        or not lower_limit <= initial_position <= upper_limit
    ):
        raise ValueError("Invalid actuator arrays, times or limits")
    positions = [float(initial_position)]
    # Exact integration of dx/dt=clip((target-x)/tau, -rate, rate), including rate-to-exponential transition.
    for dt, target in zip(np.diff(t), np.clip(u[:-1], lower_limit, upper_limit)):
        x = positions[-1]
        error = target - x
        sign = np.sign(error)
        distance = abs(error)
        threshold = rate * tau
        if distance > threshold:
            rate_duration = (distance - threshold) / rate
            if dt <= rate_duration:
                x += sign * rate * dt
            else:
                x = target - sign * threshold * np.exp(-(dt - rate_duration) / tau)
        else:
            x = target - error * np.exp(-dt / tau)
        positions.append(float(np.clip(x, lower_limit, upper_limit)))
    position = np.array(positions)
    return _out(
        "Single first-order actuator with exact rate-limited interval integration and held commands; no load-dependent torque, backlash or hysteresis.",
        position=position,
        tracking_error=u - position,
        command_clipped=(u < lower_limit) | (u > upper_limit),
        average_rate_per_s=np.diff(position) / np.diff(t),
    )


def hold_down_loads(
    weight_n,
    thrust_n,
    lateral_force_n,
    overturning_moment_nm,
    support_radius_m,
    support_count,
):
    """Rigid equally spaced circular hold-down vertical reactions; N, N*m, m. Applied overturning about one diameter."""
    weight, thrust = _positive([weight_n, thrust_n], "weight/thrust", zero=True)
    radius = float(_positive(support_radius_m, "support_radius_m"))
    _array([lateral_force_n, overturning_moment_nm], "lateral force/moment")
    if not isinstance(support_count, int) or support_count < 3:
        raise ValueError("At least three equally spaced supports required")
    x = radius * np.cos(np.arange(support_count) * 2 * np.pi / support_count)
    reactions = (weight - thrust) / support_count + overturning_moment_nm * x / (x @ x)
    return _out(
        "Rigid ring with identical vertical stiffness; signed vertical reaction positive in compression, negative in tension; lateral shear evenly shared. Local stress, compliance and release dynamics excluded.",
        vertical_reaction_n=reactions,
        shear_per_support_n=lateral_force_n / support_count,
        peak_tension_n=max(0.0, float(-reactions.min())),
        peak_compression_n=max(0.0, float(reactions.max())),
    )


FUNCTIONS = {
    f"systems_{name}": fn
    for name, fn in {
        "rf_link": rf_link_budget,
        "satellite_visibility": satellite_visibility,
        "telemetry": telemetry_budget,
        "power_energy": power_energy,
        "dc_circuit": dc_circuit,
        "timing": timing_budget,
        "state_machine": state_machine,
        "launch_commit": launch_commit_rules,
        "navigation": navigation_covariance,
        "frequency_margins": frequency_margins,
        "linear_feedback": linear_feedback,
        "fault_tree": fault_tree,
        "fmea": fmea_ranking,
        "critical_path": critical_path,
        "cost": parametric_cost,
        "transport_clearance": transport_clearance,
        "lift": lift_reactions,
        "tolerance": tolerance_stack,
        "concentration": concentration_balance,
        "humidity": humidity_balance,
        "thermal": thermal_response,
        "weather_separation": weather_separation,
        "pressure_trend": pressure_trend,
        "actuator": actuator_response,
        "hold_down": hold_down_loads,
    }.items()
}


# Exact source wording only. A kernel addresses the named analytical activity,
# subject to its documented model scope. Physical tests/inspection remain unmapped.
_ACTIVITIES = {
    "rf_link": ["RF link budget and antenna-visibility sweep."],
    "satellite_visibility": ["GNSS satellite-visibility geometry sweep."],
    "telemetry": [
        "Telemetry packet-rate and bandwidth budget.",
        "Engine-command telemetry packet-rate budget.",
        "Ground communication bandwidth and latency budgets.",
    ],
    "power_energy": [
        "Electrical load and energy balance.",
        "Charging and operating electrical-load balance.",
        "Backup-power energy and transfer-time budgets.",
        "Ground electrical load-balance calculations.",
    ],
    "dc_circuit": [
        "Wire ampacity, voltage-drop and fault-current calculations.",
        "Circuit fault-current and protection-clearing calculations.",
        "Ground power-load and voltage-drop calculations.",
    ],
    "timing": ["Umbilical release timing-budget calculations."],
    "state_machine": [
        "Mechanism state-transition and command-timing model.",
        "Separation state-transition and event-timing model.",
        "Termination-command state-transition model.",
        "Separation-command state-transition evaluation.",
        "Countdown state-transition table evaluation.",
    ],
    "launch_commit": [
        "Precipitation-rule truth-table evaluation.",
        "Disturbed-weather rule truth-table evaluation.",
        "Lightning-criteria truth-table evaluation.",
    ],
    "navigation": [
        "Sensor-error propagation and estimator covariance model.",
        "Sensor bias and noise error-budget propagation.",
        "Navigation error propagation and Monte Carlo dispersions.",
    ],
    "linear_feedback": ["Closed-loop state-feedback simulation."],
    "fault_tree": [
        "Hazard fault-tree analysis.",
        "Communication failure-mode and fault-tree evaluation.",
    ],
    "fmea": [
        "Failure modes and effects analysis (FMEA).",
        "Battery FMEA and thermal-runaway fault tree.",
        "Pad-hazard FMEA and fault-tree evaluation.",
    ],
    "critical_path": ["Critical-path program scheduling."],
    "cost": [
        "Parametric development and recurring-cost model.",
        "Parametric recurring-cost and launch-price model.",
    ],
    "transport_clearance": ["Transport clearance and payload-capacity calculations."],
    "lift": ["Lift-load and rigging-force calculations."],
    "tolerance": ["Interface tolerance-stack calculations."],
    "concentration": [
        "Cleanroom particle concentration-balance calculations.",
        "Particle concentration-balance calculations.",
    ],
    "humidity": ["Payload moisture mass-balance calculations."],
    "thermal": [
        "Ambient-temperature thermal-response calculations.",
        "Connector transient heat-transfer calculations.",
    ],
    "weather_separation": [
        "Lightning distance and elapsed-time calculations.",
        "Storm-to-flight-path separation-distance calculations.",
    ],
    "actuator": ["Gimbal and throttle actuator-response simulation."],
    "hold_down": ["Hold-down reaction-load calculations."],
}
KERNEL_ACTIVITIES = {
    activity: f"systems_{key}"
    for key, values in _ACTIVITIES.items()
    for activity in values
}


def power_network(energy_inputs, circuit_inputs):
    """Combined energy/duration and DC terminal-voltage budget; nested input contracts are power_energy and dc_circuit."""
    return _out(
        "Two independent design-budget models; constant-load energy and resistive DC circuit assumptions apply.",
        energy=power_energy(**energy_inputs),
        circuit=dc_circuit(**circuit_inputs),
    )


def state_logic(state_inputs, timing_inputs, fault_inputs):
    """Combined deterministic event/interlock sequence, worst-case timing and independent-event fault tree; nested contracts explicit."""
    return _out(
        "Supplied state table/cases, serial timing bounds and independent event probabilities; not exhaustive formal verification or a hardware test.",
        sequence=state_machine(**state_inputs),
        timing=timing_budget(**timing_inputs),
        faults=fault_tree(**fault_inputs),
    )


def safety_decision(rule_inputs, fault_inputs):
    """Supplied launch predicates plus independent-event hazard fault tree; does not authorize launch."""
    return _out(
        "Analyst-sourced rules and fault tree; no regulatory policy or default go criteria.",
        decision=launch_commit_rules(**rule_inputs),
        faults=fault_tree(**fault_inputs),
    )


def rf_telemetry(link_inputs, telemetry_inputs):
    """Combined free-space RF/visibility and packet bandwidth/latency budgets; nested contracts rf_link_budget and telemetry_budget."""
    if link_inputs["bit_rate_bps"] != telemetry_inputs["link_rate_bps"]:
        raise ValueError("RF bit rate and telemetry channel bit rate must agree")
    return _out(
        "Free-space RF plus steady-rate packet budget; coding gain belongs in supplied required Eb/N0.",
        link=rf_link_budget(**link_inputs),
        telemetry=telemetry_budget(**telemetry_inputs),
    )


def handling_analysis(clearance_inputs, lift_inputs, tolerance_inputs):
    """Transport/mating/handling envelope, two-point support loads and tolerance stack; nested contracts explicit, m/kg/N."""
    if clearance_inputs["payload_mass_kg"] != lift_inputs["mass_kg"]:
        raise ValueError("Transport and lift masses must agree")
    return _out(
        "Axis-aligned envelope and two-point rigid support screen; actual CAD swept path, joint stress and local deflection need external evidence.",
        clearance=transport_clearance(**clearance_inputs),
        lifting=lift_reactions(**lift_inputs),
        tolerance=tolerance_stack(**tolerance_inputs),
    )


def ground_hydraulics(
    network_inputs,
    transfer_mass_kg,
    storage_capacity_kg,
    inventory_kg,
    flow_direction,
    density_kg_m3,
    wave_speed_m_s,
    velocity_change_m_s,
    permitted_pressure_pa,
):
    """Ground fill/drain time, storage inventory and Darcy pressure loss plus instantaneous Joukowsky surge; SI units."""
    from .propulsion import hydraulic_network

    transfer, capacity, inventory = _positive(
        [transfer_mass_kg, storage_capacity_kg, inventory_kg], "inventories", zero=True
    )
    density, wave = _positive([density_kg_m3, wave_speed_m_s], "density/wave speed")
    _array(velocity_change_m_s, "velocity_change_m_s")
    limit = float(_positive(permitted_pressure_pa, "permitted_pressure_pa"))
    flow = float(_positive(network_inputs["mass_flow_kg_s"], "mass_flow_kg_s"))
    if network_inputs["density_kg_m3"] != density_kg_m3:
        raise ValueError("Surge and network densities must agree")
    if flow_direction not in ("fill_vehicle", "drain_vehicle"):
        raise ValueError("flow_direction must be fill_vehicle or drain_vehicle")
    final = (
        inventory - transfer
        if flow_direction == "fill_vehicle"
        else inventory + transfer
    )
    surge = density * wave * abs(velocity_change_m_s)
    network = hydraulic_network(**network_inputs)
    return _out(
        "Steady incompressible Darcy flow at imposed rate and instantaneous closure Joukowsky surge bound. No cavitation, two-phase cryogenic flow or valve transients.",
        network=network,
        transfer_time_s=transfer / flow,
        final_storage_inventory_kg=final,
        storage_capacity_margin_kg=capacity - final,
        inventory_nonnegative=bool(final >= 0),
        surge_pressure_pa=surge,
        peak_pressure_screen_pa=network_inputs["upstream_pressure_Pa"] + surge,
        pressure_margin_pa=limit - network_inputs["upstream_pressure_Pa"] - surge,
    )


def wind_strip_loads(
    height_m,
    wind_speed_m_s,
    density_kg_m3,
    projected_area_m2,
    drag_coefficient,
    section_modulus_m3,
    allowable_stress_pa,
):
    """Static wind-strip force, base/shear bending and section stress; height m, velocity m/s, rho kg/m^3, areas m^2, Z m^3, stress Pa."""
    h, v, area, cd, z = (
        _array(x, n, 1)
        for x, n in [
            (height_m, "height_m"),
            (wind_speed_m_s, "wind_speed_m_s"),
            (projected_area_m2, "projected_area_m2"),
            (drag_coefficient, "drag_coefficient"),
            (section_modulus_m3, "section_modulus_m3"),
        ]
    )
    if (
        any(a.shape != h.shape for a in [v, area, cd, z])
        or np.any(h < 0)
        or np.any(np.diff(h) <= 0)
    ):
        raise ValueError(
            "Equal-length arrays and nonnegative increasing heights required"
        )
    _positive(area, "areas", zero=True)
    _positive(cd, "coefficients", zero=True)
    _positive(z, "section modulus")
    rho = float(_positive(density_kg_m3, "density"))
    allowable = float(_positive(allowable_stress_pa, "allowable stress"))
    q = 0.5 * rho * v**2
    force = q * area * cd * np.sign(v)
    moments = np.array(
        [sum(f * (x - cut) for f, x in zip(force, h) if x >= cut) for cut in h]
    )
    stress = np.abs(moments) / z
    return _out(
        "Quasi-static cantilever strip loads in one lateral plane; supplied drag coefficients/section properties, no gust dynamics, aerodynamic interference, buckling or combined axial stress.",
        dynamic_pressure_pa=q,
        strip_force_n=force,
        base_shear_n=float(force.sum()),
        base_moment_nm=float(force @ h),
        section_moment_nm=moments,
        bending_stress_pa=stress,
        stress_margin_pa=allowable - stress,
    )


def radiation_dose(
    time_s,
    energy_mev,
    differential_flux_m2_s_mev,
    shielding_transfer_matrix,
    dose_response_gy_m2,
    allowable_dose_gy,
):
    """Transport using a supplied spectral response matrix then integrate dose; flux /m^2/s/MeV, energy MeV, response Gy*m^2."""
    time = _array(time_s, "time_s", 1)
    energy = _positive(energy_mev, "energy_mev")
    flux = _positive(differential_flux_m2_s_mev, "flux", zero=True)
    matrix = _positive(
        shielding_transfer_matrix, "shielding_transfer_matrix", zero=True
    )
    response = _positive(dose_response_gy_m2, "response", zero=True)
    limit = float(_positive(allowable_dose_gy, "allowable_dose_gy"))
    if (
        len(time) < 2
        or len(energy) < 2
        or np.any(np.diff(time) <= 0)
        or np.any(np.diff(energy) <= 0)
        or flux.shape != (len(time), len(energy))
        or matrix.shape != (len(energy), len(energy))
        or response.shape != energy.shape
    ):
        raise ValueError("Time/energy and spectral matrix dimensions are incompatible")
    # Matrix maps input differential-flux samples to output differential-flux samples;
    # response is dose per unit particle fluence at each sampled output energy.
    transported = flux @ matrix.T
    dose_rate = trapezoid(transported * response, energy, axis=1)
    dose = float(trapezoid(dose_rate, time))
    return _out(
        "Linear spectral transfer matrix and dose response must come from validated transport/material data; this kernel does not derive shielding, particle interactions or single-event upset rates.",
        dose_rate_gy_s=dose_rate,
        total_dose_gy=dose,
        dose_margin_gy=limit - dose,
    )


def rolling_sphere(mast_height_m, sphere_radius_m, object_distance_m, object_height_m):
    """Single isolated mast rolling-sphere protected envelope over flat ground; dimensions m, heights <= sphere radius."""
    h, r = _positive([mast_height_m, sphere_radius_m], "mast/radius")
    d, z = _positive(object_distance_m, "distances", zero=True), _positive(
        object_height_m, "heights", zero=True
    )
    if d.shape != z.shape or h > r or np.any(z > h):
        raise ValueError(
            "Equal object arrays, object heights<=mast<=sphere radius required"
        )
    boundary = np.sqrt(2 * r * h - h * h) - np.sqrt(2 * r * z - z * z)
    return _out(
        "Geometric single-mast rolling sphere over level ground, ideal point mast. Sphere radius is externally specified; no lightning-risk certification, multi-mast shielding or bonding assessment.",
        protection_radius_m=boundary,
        radial_margin_m=boundary - d,
        inside_modeled_envelope=d <= boundary,
    )


def environment_balance(
    thermal_inputs,
    humidity_inputs,
    particle_inputs,
    heat_load_w,
    supply_temperature_k,
    maximum_temperature_k,
    air_density_kg_m3,
    air_heat_capacity_j_kg_k,
):
    """Combined lumped thermal, moisture/particle balances and steady sensible-heat conditioning airflow; SI units."""
    heat = float(_positive(heat_load_w, "heat_load_w", zero=True))
    rho, cp = _positive([air_density_kg_m3, air_heat_capacity_j_kg_k], "air properties")
    _positive([supply_temperature_k, maximum_temperature_k], "temperatures")
    if maximum_temperature_k <= supply_temperature_k:
        raise ValueError("Cooling limit must exceed supply temperature")
    return _out(
        "Mixed-volume/lumped thermal screen with constant air properties; no local contamination, condensation heat, thermal gradients or flow distribution.",
        thermal=thermal_response(**thermal_inputs),
        humidity=humidity_balance(**humidity_inputs),
        particles=concentration_balance(**particle_inputs),
        required_sensible_airflow_m3_s=heat
        / (rho * cp * (maximum_temperature_k - supply_temperature_k)),
    )


def schedule_resources(
    tasks, resource_rates_per_day, initial_inventory, fixed_consumption
):
    """Critical-path schedule and resource inventory; common amount units per resource, rates amount/day, task durations days."""
    result = critical_path(tasks)
    names = (
        set(resource_rates_per_day) | set(initial_inventory) | set(fixed_consumption)
    )
    if not (
        names
        == set(resource_rates_per_day)
        == set(initial_inventory)
        == set(fixed_consumption)
    ):
        raise ValueError("Resource keys must agree")
    consumption = {}
    remaining = {}
    for name in sorted(names):
        rate, initial, fixed = _positive(
            [
                resource_rates_per_day[name],
                initial_inventory[name],
                fixed_consumption[name],
            ],
            name,
            zero=True,
        )
        consumption[name] = float(rate * result["duration_days"] + fixed)
        remaining[name] = float(initial - consumption[name])
    return _out(
        "Deterministic schedule and continuous resource consumption throughout elapsed critical-path duration; no replenishment or resource-constrained scheduling.",
        schedule=result,
        consumption=consumption,
        remaining_inventory=remaining,
    )


def loop_response(feedback_inputs, frequency_inputs, cases_initial_state):
    """Linear closed-loop eigenvalues/recovery sensitivity and sampled SISO margins; nested contracts plus initial-state cases."""
    nominal = linear_feedback(**feedback_inputs)
    cases = []
    for state in cases_initial_state:
        args = dict(feedback_inputs)
        args["initial_state"] = state
        run = linear_feedback(**args)
        cases.append(
            dict(
                initial_state=state,
                final_state=run["state"][-1],
                peak_abs_command=np.max(np.abs(run["command"]), axis=0),
            )
        )
    return _out(
        "Analyst must provide consistent plant/K and open-loop frequency data including flexible modes if relevant. Linear unsaturated recovery only; not nonlinear trajectory recovery.",
        nominal=nominal,
        margins=frequency_margins(**frequency_inputs),
        cases=cases,
    )


def risk_scores(risks):
    """Cost/schedule risk expected-exposure ranking; records id, probability, cost_impact, delay_days in common currency and days."""
    if not risks:
        raise ValueError("Risks required")
    rows = []
    for risk in risks:
        p = float(_fraction(risk["probability"], "probability"))
        cost, days = _positive(
            [risk["cost_impact"], risk["delay_days"]], "impacts", zero=True
        )
        rows.append(
            dict(id=risk["id"], expected_cost=p * cost, expected_delay_days=p * days)
        )
    return _out(
        "Supplied marginal probabilities and consequence estimates; sum of delay exposure is not a schedule completion distribution or critical-path delay.",
        risks=rows,
        total_expected_cost=sum(r["expected_cost"] for r in rows),
        sum_expected_delay_exposure_days=sum(r["expected_delay_days"] for r in rows),
    )


def pressure_forecast_check(trend_inputs, forecast_pressure_pa, observed_pressure_pa):
    """Pressure trend plus paired forecast-observation RMSE/bias/correlation; pressures Pa, trend times s."""
    f, o = _positive(forecast_pressure_pa, "forecast_pressure_pa"), _positive(
        observed_pressure_pa, "observed_pressure_pa"
    )
    if f.ndim != 1 or f.shape != o.shape or len(f) < 3:
        raise ValueError("At least three paired forecasts/observations required")
    error = f - o
    correlation = (
        float(np.corrcoef(f, o)[0, 1]) if np.std(f) > 0 and np.std(o) > 0 else None
    )
    return _out(
        "Paired retrospective forecast validation and linear local pressure trend; no causal weather or safety prediction.",
        trend=pressure_trend(**trend_inputs),
        forecast_bias_pa=float(error.mean()),
        forecast_rmse_pa=float(np.sqrt(np.mean(error**2))),
        correlation=correlation,
    )


def swept_handling(clearance_inputs, vehicle_yaw_deg, lift_inputs, tolerance_inputs):
    """Sampled yaw-rotated rigid envelope and two-support axle/load screen; clearance/lift contracts plus yaw degrees per route box."""
    base = handling_analysis(clearance_inputs, lift_inputs, tolerance_inputs)
    yaw = _array(vehicle_yaw_deg, "vehicle_yaw_deg", 1)
    route = _array(clearance_inputs["route_envelopes_m"], "route_envelopes_m", 2)
    if yaw.shape != (len(route),):
        raise ValueError("One yaw angle per sampled route envelope required")
    width, height, length = clearance_inputs["vehicle_envelope_m"]
    sine, cosine = np.abs(np.sin(np.deg2rad(yaw))), np.abs(np.cos(np.deg2rad(yaw)))
    rotated = np.column_stack(
        (
            width * cosine + length * sine,
            np.full(len(yaw), height),
            length * cosine + width * sine,
        )
    )
    margin = route - rotated - 2 * clearance_inputs["clearance_allowance_m"]
    return _out(
        "Sampled yaw-oriented bounding boxes centered in supplied route boxes; two-point static axle/load reactions. No interpolation between poses, off-center paths, pitch/roll, articulated trailer geometry or tire dynamics.",
        sampled_rotated_envelope_m=rotated,
        clearance_margin_m=margin,
        minimum_clearance_margin_m=float(margin.min()),
        lifting=base["lifting"],
        tolerance=base["tolerance"],
        capacity_margin_kg=base["clearance"]["capacity_margin_kg"],
    )


FUNCTIONS.update(
    {
        f"systems_{key}": fn
        for key, fn in {
            "power_network": power_network,
            "state_logic": state_logic,
            "safety_decision": safety_decision,
            "rf_telemetry": rf_telemetry,
            "handling": handling_analysis,
            "ground_hydraulics": ground_hydraulics,
            "wind_loads": wind_strip_loads,
            "radiation": radiation_dose,
            "rolling_sphere": rolling_sphere,
            "environment": environment_balance,
            "schedule_resources": schedule_resources,
            "loop_response": loop_response,
            "risk_scores": risk_scores,
            "pressure_forecast": pressure_forecast_check,
            "swept_handling": swept_handling,
        }.items()
    }
)

_COMPOSITE_ACTIVITIES = {
    "power_network": [
        "Electrical energy balance and terminal voltage-drop calculations."
    ],
    "state_logic": [
        "Abort state-transition and fault-tree models.",
        "Flight-safety state-transition and fault-tree models.",
        "Automatic-hold fault-tree and state-transition evaluation.",
        "Termination-command fault-tree and state-transition evaluation.",
        "Countdown state-transition and timing-budget evaluation.",
        "Link-recovery state-transition and timing model.",
        "Termination-command state-transition and latency model.",
        "Safing-state transition and failure-mode evaluation.",
    ],
    "safety_decision": ["Hazard fault-tree and launch-commit decision models."],
    "rf_link": [
        "RF link budget and station-visibility geometry.",
        "Ground-station visibility sweep.",
    ],
    "rf_telemetry": ["Telemetry bandwidth and RF link budgets."],
    "telemetry": ["Packet-rate and end-to-end latency budget."],
    "dc_circuit": ["Circuit voltage-drop calculations."],
    "state_machine": ["Fairing-release state-transition model."],
    "frequency_margins": ["Linearized-loop gain and phase-margin calculations."],
    "handling": [
        "CAD access-clearance and support-load calculations.",
        "CAD fairing-closure and handling-clearance calculations.",
        "CAD handling-clearance and lift-load calculations.",
        "CAD payload-mating clearance and lift-load calculations.",
        "CAD clearance and tolerance-stack calculations.",
    ],
    "swept_handling": [
        "Rollout swept-clearance and support-load calculations.",
        "Transport swept-clearance and axle-load calculations.",
        "Transport swept-envelope and clearance calculations.",
    ],
    "ground_hydraulics": [
        "Cold-flow pressure-loss and resource-budget calculations.",
        "Fill-time and hydraulic pressure-loss calculations.",
        "Drain-time and return-flow hydraulic calculations.",
        "Fill/drain hydraulic and storage-capacity calculations.",
        "Fluid-network pressure and surge calculations.",
    ],
    "wind_loads": [
        "Surface-wind load and threshold-sensitivity calculations.",
        "Wind-profile aerodynamic load and stress calculations.",
    ],
    "radiation": ["Radiation transport and accumulated-dose calculation."],
    "rolling_sphere": ["Rolling-sphere lightning-protection coverage calculations."],
    "environment": [
        "Payload-volume heat-balance and airflow calculations.",
        "Transport thermal and moisture exposure calculations.",
        "Transport-container transient heat and moisture balances.",
    ],
    "thermal": [
        "Equipment transient heat-balance simulation.",
        "Payload transient heat-balance simulation.",
        "Transient heat-balance simulation.",
        "Payload-compartment transient heat-balance simulation.",
    ],
    "schedule_resources": [
        "Scrub-recovery critical-path and consumable-budget calculations.",
        "Rehearsal timing and consumable-budget calculations.",
        "Scrub-recycle timeline and consumables balance.",
    ],
    "loop_response": [
        "Flexible-mode closed-loop eigenvalue and frequency-response calculations.",
        "Attitude-offset recovery sensitivity sweep.",
        "Flight-path deviation recovery sensitivity sweep.",
    ],
    "risk_scores": ["Cost and schedule risk scoring."],
    "pressure_forecast": [
        "Pressure-trend time-series and forecast-correlation calculations."
    ],
}
KERNEL_ACTIVITIES.update(
    {
        activity: f"systems_{key}"
        for key, values in _COMPOSITE_ACTIVITIES.items()
        for activity in values
    }
)
