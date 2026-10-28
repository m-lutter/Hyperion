"""Structural and thermal calculations with supplied loads and properties.

Use section/tank functions for local stresses, FE functions for supplied matrix
models, and thermal functions for temperatures and clearances. Inputs and result
keys include SI units. Load factors and material allowables must be supplied.

These are calculation tools, not qualification evidence. Geometry, material
allowables, load factors, boundary conditions and model matrices must be supplied.
Linear isotropic and small-deflection assumptions are explicit; nonlinear FEA,
manufacturing imperfections, test correlation and mesh convergence are external.
Activity mappings identify applicable kernels, not automatically closed requirements.

References: NASA SP-8007 Rev 2 https://ntrs.nasa.gov/citations/20205011530
(shell imperfection sensitivity; no universal knockdown factor is assumed),
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html,
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.linalg import eigh, eigvalsh, eig
from scipy.signal import lsim, StateSpace


def _a(value, name, ndim=None):
    out = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(out)) or (ndim is not None and out.ndim != ndim):
        raise ValueError(
            f"{name}: finite {ndim if ndim is not None else ''}-dimensional data required"
        )
    return out


def _pos(value, name, zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return value


def _factor(value, name="load_factor"):
    value = _pos(value, name)
    if value < 1:
        raise ValueError(f"{name} must be at least one")
    return value


def _matrix(value, name, n=None, positive=False):
    a = _a(value, name, 2)
    if (
        not a.shape[0]
        or a.shape[0] != a.shape[1]
        or (n is not None and a.shape != (n, n))
    ):
        raise ValueError(f"{name} must be a square matrix of the required size")
    if not np.allclose(a, a.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if positive and np.min(eigvalsh(a)) <= 0:
        raise ValueError(
            f"{name} must be positive definite after boundary-condition reduction"
        )
    return a


def _times(times_s):
    t = _a(times_s, "times_s", 1)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("at least two strictly increasing times required")
    return t


def _history(value, t, n, name):
    a = _a(value, name)
    if a.ndim == 0:
        a = np.full((len(t), n), float(a))
    elif a.shape == (n,):
        a = np.tile(a, (len(t), 1))
    elif n == 1 and a.shape == (len(t),):
        a = a[:, None]
    if a.shape != (len(t), n):
        raise ValueError(
            f"{name} requires a constant vector ({n},) or history ({len(t)},{n})"
        )
    return a


def _interp(t, history, time):
    return np.array(
        [np.interp(time, t, history[:, j]) for j in range(history.shape[1])]
    )


def _margin(demand, allowable):
    demand = _pos(demand, "demand", zero=True)
    allowable = _pos(allowable, "allowable")
    return {
        "demand": demand,
        "allowable": allowable,
        "utilization": demand / allowable,
        "margin": allowable / demand - 1 if demand > 0 else None,
        "pass": bool(demand <= allowable),
    }


def von_mises(stress_pa):
    """Rows ordered xx, yy, zz, xy, yz, zx; shear entries are stress, not strain."""
    s = _a(stress_pa, "stress_pa")
    if s.shape[-1] != 6:
        raise ValueError("stress tensor requires six components")
    return np.sqrt(
        0.5
        * (
            (s[..., 0] - s[..., 1]) ** 2
            + (s[..., 1] - s[..., 2]) ** 2
            + (s[..., 2] - s[..., 0]) ** 2
        )
        + 3 * np.sum(s[..., 3:] ** 2, axis=-1)
    )


def section_stress(
    *,
    axial_force_n,
    bending_y_nm,
    bending_z_nm,
    shear_y_n,
    shear_z_n,
    torque_nm,
    area_m2,
    iy_m4,
    iz_m4,
    torsion_j_m4,
    shear_area_y_m2,
    shear_area_z_m2,
    sample_yz_m,
    yield_allowable_pa,
    ultimate_allowable_pa,
    yield_factor,
    ultimate_factor,
):
    """Prismatic section about principal centroidal axes; Saint-Venant circular torsion.

    sample_yz_m must include all critical section points. Shear areas are supplied
    effective peak-stress areas, not automatically inferred from section area.
    Signs: sigma_x=N/A-My*z/Iy+Mz*y/Iz, tau_xy=Vy/Ay-T*z/J.
    """
    area = _pos(area_m2, "area_m2")
    iy = _pos(iy_m4, "iy_m4")
    iz = _pos(iz_m4, "iz_m4")
    torsion_constant = _pos(torsion_j_m4, "torsion_j_m4")
    shear_area_y = _pos(shear_area_y_m2, "shear_area_y_m2")
    shear_area_z = _pos(shear_area_z_m2, "shear_area_z_m2")
    sample_points = _a(sample_yz_m, "sample_yz_m", 2)
    if sample_points.shape[1] != 2 or not len(sample_points):
        raise ValueError("sample_yz_m must contain critical y,z section coordinates")
    axial_force, bending_y, bending_z, shear_y, shear_z, torque = _a(
        [axial_force_n, bending_y_nm, bending_z_nm, shear_y_n, shear_z_n, torque_nm],
        "loads",
    )
    y, z = sample_points.T
    stress = np.zeros((len(sample_points), 6))
    stress[:, 0] = axial_force / area - bending_y * z / iy + bending_z * y / iz
    stress[:, 3] = shear_y / shear_area_y - torque * z / torsion_constant
    stress[:, 5] = shear_z / shear_area_z + torque * y / torsion_constant
    equivalent_stress = von_mises(stress)
    return {
        "stress_pa": stress.tolist(),
        "von_mises_pa": equivalent_stress.tolist(),
        "critical_sample": int(np.argmax(equivalent_stress)),
        "yield": _margin(
            float(equivalent_stress.max()) * _factor(yield_factor), yield_allowable_pa
        ),
        "ultimate_equivalent_stress_screen": _margin(
            float(equivalent_stress.max()) * _factor(ultimate_factor),
            ultimate_allowable_pa,
        ),
        "scope": "Linear isotropic section stress. Ultimate equivalent stress is a screening calculation, not a fracture or joint failure criterion.",
    }


def tank_stress(
    *,
    radius_m,
    thickness_m,
    pressure_pa,
    axial_force_n,
    bending_moment_nm,
    torque_nm,
    yield_allowable_pa,
    joint_efficiency,
    load_factor,
):
    """Thin closed-end cylinder membrane plus axial bending; extrema at both fibers.

    Pressure is differential (can be negative). External-pressure instability must
    be checked separately. Requires t/r <= 0.1; excludes local discontinuity stress.
    """
    radius = _pos(radius_m, "radius_m")
    thickness = _pos(thickness_m, "thickness_m")
    efficiency = _pos(joint_efficiency, "joint_efficiency")
    if thickness / radius > 0.1 or efficiency > 1:
        raise ValueError("thin-shell t/r<=0.1 and joint_efficiency<=1 required")
    pressure, axial_force, bending_moment, torque = _a(
        [pressure_pa, axial_force_n, bending_moment_nm, torque_nm], "loads"
    )
    axial = axial_force / (2 * np.pi * radius * thickness) + pressure * radius / (
        2 * thickness
    )
    bending = abs(bending_moment) / (np.pi * radius * radius * thickness)
    hoop = pressure * radius / thickness
    shear = torque / (2 * np.pi * radius * radius * thickness)
    stress = np.array(
        [
            [axial - bending, hoop, 0, shear, 0, 0],
            [axial + bending, hoop, 0, shear, 0, 0],
        ]
    )
    equivalent_stress = von_mises(stress)
    return {
        "stress_pa": stress.tolist(),
        "von_mises_pa": equivalent_stress.tolist(),
        "maximum_axial_compression_pa": max(0.0, -float(stress[:, 0].min())),
        "yield": _margin(
            equivalent_stress.max() * _factor(load_factor),
            _pos(yield_allowable_pa, "yield_allowable_pa") * efficiency,
        ),
        "scope": "Thin isotropic cylindrical membrane; no penetration, weld-detail, plasticity, or buckling solution.",
    }


def buckling(
    *,
    youngs_modulus_pa,
    poisson_ratio,
    radius_m,
    thickness_m,
    length_m,
    area_m2,
    min_second_moment_m4,
    effective_length_factor,
    shell_knockdown,
    column_knockdown,
    compressive_force_n,
    load_factor,
):
    """Independent ideal axial shell and Euler-column screens with supplied KDFs.

    compressive_force_n must represent the destabilizing load, including bending
    as appropriate. No stabilizing pressure credit or combined-mode interaction.
    """
    youngs_modulus = _pos(youngs_modulus_pa, "youngs_modulus_pa")
    poisson = float(poisson_ratio)
    if not -1 < poisson < 0.5:
        raise ValueError("isotropic Poisson ratio must lie in (-1,.5)")
    radius = _pos(radius_m, "radius_m")
    thickness = _pos(thickness_m, "thickness_m")
    length = _pos(length_m, "length_m")
    area = _pos(area_m2, "area_m2")
    second_moment = _pos(min_second_moment_m4, "min_second_moment_m4")
    effective_length = _pos(effective_length_factor, "effective_length_factor")
    shell_factor = _pos(shell_knockdown, "shell_knockdown")
    column_factor = _pos(column_knockdown, "column_knockdown")
    if max(shell_factor, column_factor) > 1 or thickness / radius > 0.1:
        raise ValueError("KDFs <=1 and thin shell t/r<=.1 required")
    shell_stress = (
        shell_factor
        * youngs_modulus
        * thickness
        / radius
        / math.sqrt(3 * (1 - poisson * poisson))
    )
    column_load = (
        column_factor
        * math.pi**2
        * youngs_modulus
        * second_moment
        / (effective_length * length) ** 2
    )
    demand = _pos(compressive_force_n, "compressive_force_n", zero=True) * _factor(
        load_factor
    )
    return {
        "shell_critical_stress_pa": shell_stress,
        "shell_critical_load_n": shell_stress * area,
        "column_critical_load_n": column_load,
        "shell": _margin(demand, shell_stress * area),
        "column": _margin(demand, column_load),
        "scope": "Axial ideal isotropic shell and elastic column screens; externally supplied KDFs, no pressure stabilization credit.",
    }


def buckling_eigenvalue(
    *, stiffness_n_m, destabilizing_geometric_stiffness_n_m, load_factor
):
    """K v = lambda G v; G is supplied for one reference destabilizing load case."""
    stiffness = _matrix(stiffness_n_m, "stiffness_n_m", positive=True)
    geometric_stiffness = _matrix(
        destabilizing_geometric_stiffness_n_m, "geometric_stiffness", len(stiffness)
    )
    eigenvalues = eig(stiffness, geometric_stiffness, right=False)
    positive = sorted(
        float(v.real)
        for v in eigenvalues
        if np.isfinite(v) and abs(v.imag) < 1e-8 * max(1, abs(v.real)) and v.real > 0
    )
    if not positive:
        raise ValueError(
            "no positive finite buckling eigenvalue for supplied geometric stiffness"
        )
    factor = _factor(load_factor)
    return {
        "critical_load_multipliers": positive,
        "minimum_multiplier": positive[0],
        "margin": positive[0] / factor - 1,
        "pass": positive[0] >= factor,
        "scope": "Linear eigenvalue bifurcation only; imperfection/nonlinear correlation required.",
    }


def joint_strength(
    *,
    fastener_tension_n,
    fastener_shear_n,
    fastener_tensile_area_m2,
    fastener_shear_area_m2,
    bearing_diameter_m,
    plate_thickness_m,
    tension_allowable_pa,
    shear_allowable_pa,
    bearing_allowable_pa,
    weld_force_n,
    weld_effective_throat_area_m2,
    weld_allowable_pa,
    preload_n,
    joint_stiffness_fraction,
    load_factor,
):
    """Per-fastener loads from an external load distribution; equal load not assumed.

    Elastic bolt load = preload + C*external tension; clamp remaining = preload-
    (1-C)*external tension. Bolt interaction is quadratic tension/shear screening.
    """
    tension = _pos(fastener_tension_n, "fastener_tension_n", True)
    shear = _pos(fastener_shear_n, "fastener_shear_n", True)
    preload = _pos(preload_n, "preload_n", True)
    c = float(joint_stiffness_fraction)
    if not 0 <= c <= 1:
        raise ValueError("joint_stiffness_fraction must be [0,1]")
    f = _factor(load_factor)
    sigma = (preload + c * f * tension) / _pos(
        fastener_tensile_area_m2, "fastener_tensile_area_m2"
    )
    tau = f * shear / _pos(fastener_shear_area_m2, "fastener_shear_area_m2")
    bearing = (
        f
        * shear
        / (
            _pos(bearing_diameter_m, "bearing_diameter_m")
            * _pos(plate_thickness_m, "plate_thickness_m")
        )
    )
    rt = sigma / _pos(tension_allowable_pa, "tension_allowable_pa")
    rs = tau / _pos(shear_allowable_pa, "shear_allowable_pa")
    weld = (
        f
        * abs(float(weld_force_n))
        / _pos(weld_effective_throat_area_m2, "weld_effective_throat_area_m2")
    )
    clamp = preload - (1 - c) * f * tension
    return {
        "bolt_tension_pa": sigma,
        "bolt_shear_pa": tau,
        "bearing": _margin(bearing, bearing_allowable_pa),
        "weld": _margin(weld, weld_allowable_pa),
        "bolt_interaction": rt * rt + rs * rs,
        "bolt_pass": rt * rt + rs * rs <= 1,
        "remaining_clamp_n": clamp,
        "no_joint_separation": clamp >= 0,
        "scope": "Elastic preloaded-joint screen; excludes prying, slip, edge tearout, eccentric weld-group response, thread stripping and preload scatter.",
    }


def fe_static(
    *,
    stiffness_n_m,
    force_n,
    constrained_dofs,
    prescribed_displacement_m=None,
    stress_recovery_pa_m=None,
    stress_offset_pa=None,
):
    """Supplied assembled linear FE matrix; fixed constraints are exact elimination."""
    stiffness = _matrix(stiffness_n_m, "stiffness_n_m")
    dof_count = len(stiffness)
    force = _a(force_n, "force_n", 1)
    if force.shape != (dof_count,):
        raise ValueError("force vector size mismatch")
    original_fixed = _a(constrained_dofs, "constrained_dofs", 1)
    fixed_dofs = original_fixed.astype(int)
    if not np.array_equal(original_fixed, fixed_dofs):
        raise ValueError("constrained DOFs must be integers")
    if (
        np.any(fixed_dofs < 0)
        or np.any(fixed_dofs >= dof_count)
        or len(set(fixed_dofs)) != len(fixed_dofs)
    ):
        raise ValueError("invalid constrained DOFs")
    free_dofs = np.setdiff1d(np.arange(dof_count), fixed_dofs)
    displacement = np.zeros(dof_count)
    if prescribed_displacement_m is not None:
        values = _a(prescribed_displacement_m, "prescribed_displacement_m", 1)
        if values.shape != fixed_dofs.shape:
            raise ValueError("one prescribed displacement per constrained DOF")
        displacement[fixed_dofs] = values
    if len(free_dofs):
        reduced = _matrix(
            stiffness[np.ix_(free_dofs, free_dofs)], "reduced stiffness", positive=True
        )
        displacement[free_dofs] = np.linalg.solve(
            reduced,
            force[free_dofs]
            - stiffness[np.ix_(free_dofs, fixed_dofs)] @ displacement[fixed_dofs],
        )
    reaction = stiffness @ displacement - force
    result = {
        "displacement_m": displacement.tolist(),
        "reaction_n": reaction.tolist(),
        "free_equilibrium_residual_n": float(np.linalg.norm(reaction[free_dofs])),
        "strain_energy_j": float(0.5 * displacement @ stiffness @ displacement),
    }
    if stress_recovery_pa_m is not None:
        stress_recovery = _a(stress_recovery_pa_m, "stress_recovery_pa_m", 2)
        if stress_recovery.shape[1] != dof_count or stress_recovery.shape[0] % 6:
            raise ValueError("stress recovery needs 6 rows per point and n columns")
        stress = stress_recovery @ displacement
        if stress_offset_pa is not None:
            offset = _a(stress_offset_pa, "stress_offset_pa", 1)
            if offset.shape != stress.shape:
                raise ValueError("stress offset shape mismatch")
            stress += offset
        stress = stress.reshape(-1, 6)
        result.update(
            stress_pa=stress.tolist(), von_mises_pa=von_mises(stress).tolist()
        )
    result["scope"] = (
        "Linear supplied-matrix calculation; model mesh, stress recovery, constraints and correlation are caller evidence. No automatic allowable pass."
    )
    return result


def modal(*, stiffness_n_m, mass_kg):
    """Boundary-condition-reduced matrices; mass-normalized real modes."""
    stiffness = _matrix(stiffness_n_m, "stiffness_n_m")
    mass = _matrix(mass_kg, "mass_kg", len(stiffness), positive=True)
    eigenvalues, modes = eigh(stiffness, mass)
    tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * 1e-10
    if np.any(eigenvalues < -tolerance):
        raise ValueError("negative stiffness eigenvalue: unstable model")
    eigenvalues = np.maximum(eigenvalues, 0)
    return {
        "eigenvalues_rad2_s2": eigenvalues.tolist(),
        "frequency_hz": (np.sqrt(eigenvalues) / (2 * np.pi)).tolist(),
        "mass_normalized_modes": modes.tolist(),
        "mass_orthogonality_error": float(
            np.max(np.abs(modes.T @ mass @ modes - np.eye(len(mass))))
        ),
    }


def fe_transient(
    *,
    times_s,
    stiffness_n_m,
    mass_kg,
    damping_ns_m,
    force_n,
    initial_displacement_m,
    initial_velocity_m_s,
    base_acceleration_m_s2=None,
    base_influence=None,
    rtol=1e-8,
):
    """M u''+C u'+K u=f-M gamma a_base; u is relative to prescribed base.

    Linear interpolation of input histories; returns absolute acceleration when
    a base history/influence vector is supplied. Matrices already constrained.
    """
    times = _times(times_s)
    stiffness = _matrix(stiffness_n_m, "stiffness_n_m")
    dof_count = len(stiffness)
    mass = _matrix(mass_kg, "mass_kg", dof_count, True)
    damping = _matrix(damping_ns_m, "damping_ns_m", dof_count)
    if np.min(eigvalsh(stiffness)) < -1e-8 or np.min(eigvalsh(damping)) < -1e-8:
        raise ValueError("stiffness and damping must be nonnegative definite")
    force = _history(force_n, times, dof_count, "force_n")
    initial_displacement = _a(initial_displacement_m, "initial_displacement_m", 1)
    initial_velocity = _a(initial_velocity_m_s, "initial_velocity_m_s", 1)
    if initial_displacement.shape != (dof_count,) or initial_velocity.shape != (
        dof_count,
    ):
        raise ValueError("initial state shape mismatch")
    if (base_acceleration_m_s2 is None) != (base_influence is None):
        raise ValueError("base acceleration and influence must be supplied together")
    base = (
        np.zeros((len(times), 1))
        if base_acceleration_m_s2 is None
        else _history(base_acceleration_m_s2, times, 1, "base_acceleration_m_s2")
    )
    base_direction = (
        np.zeros(dof_count)
        if base_influence is None
        else _a(base_influence, "base_influence", 1)
    )
    if base_direction.shape != (dof_count,):
        raise ValueError("base influence shape mismatch")

    def rhs(time, state):
        effective_force = (
            _interp(times, force, time)
            - mass @ base_direction * _interp(times, base, time)[0]
        )
        return np.r_[
            state[dof_count:],
            np.linalg.solve(
                mass,
                effective_force
                - damping @ state[dof_count:]
                - stiffness @ state[:dof_count],
            ),
        ]

    solution = solve_ivp(
        rhs,
        (times[0], times[-1]),
        np.r_[initial_displacement, initial_velocity],
        t_eval=times,
        rtol=_pos(rtol, "rtol"),
        atol=rtol * 1e-3,
        max_step=float(np.min(np.diff(times))) / 2,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    displacement = solution.y[:dof_count].T
    velocity = solution.y[dof_count:].T
    relative_acceleration = np.array(
        [rhs(tt, yy)[dof_count:] for tt, yy in zip(times, solution.y.T)]
    )
    absolute_acceleration = relative_acceleration + base * base_direction
    return {
        "times_s": times.tolist(),
        "displacement_m": displacement.tolist(),
        "velocity_m_s": velocity.tolist(),
        "relative_acceleration_m_s2": relative_acceleration.tolist(),
        "absolute_acceleration_m_s2": absolute_acceleration.tolist(),
        "maximum_abs_acceleration_m_s2": np.max(
            np.abs(absolute_acceleration), axis=0
        ).tolist(),
        "scope": "Linear supplied-matrix transient; no nonlinear contact, yielding, or separation constraint change.",
    }


def base_psd_response(
    *,
    frequency_hz,
    base_acceleration_psd_m2_s4_hz,
    stiffness_n_m,
    mass_kg,
    damping_ns_m,
    base_influence,
):
    """One-sided base-acceleration PSD through linear multi-DOF FRF, trapezoid in Hz."""
    frequencies = _a(frequency_hz, "frequency_hz", 1)
    psd = _a(base_acceleration_psd_m2_s4_hz, "psd", 1)
    if (
        len(frequencies) < 2
        or frequencies[0] < 0
        or np.any(np.diff(frequencies) <= 0)
        or psd.shape != frequencies.shape
        or np.any(psd < 0)
    ):
        raise ValueError("increasing nonnegative frequencies and matching PSD required")
    stiffness = _matrix(stiffness_n_m, "stiffness_n_m", positive=True)
    dof_count = len(stiffness)
    mass = _matrix(mass_kg, "mass_kg", dof_count, True)
    damping = _matrix(damping_ns_m, "damping_ns_m", dof_count)
    base_direction = _a(base_influence, "base_influence", 1)
    if base_direction.shape != (dof_count,) or np.min(eigvalsh(damping)) < 0:
        raise ValueError("invalid influence or damping")
    out = []
    for f in frequencies:
        angular_frequency = 2 * np.pi * f
        response = (
            base_direction
            + angular_frequency
            * angular_frequency
            * np.linalg.solve(
                stiffness
                - angular_frequency * angular_frequency * mass
                + 1j * angular_frequency * damping,
                mass @ base_direction,
            )
        )
        out.append(np.abs(response) ** 2)
    out = np.asarray(out) * psd[:, None]
    return {
        "frequency_hz": frequencies.tolist(),
        "absolute_acceleration_psd_m2_s4_hz": out.tolist(),
        "absolute_acceleration_rms_m_s2": np.sqrt(
            trapezoid(out, frequencies, axis=0)
        ).tolist(),
        "scope": "Sampled one-sided linear spectrum; caller must resolve resonances and supply valid PSD bandwidth/correlation.",
    }


def shock_response_spectrum(
    *,
    times_s,
    acceleration_m_s2,
    natural_frequencies_hz,
    damping_ratio,
    residual_decay_constants=5,
):
    """Absolute-acceleration SRS of zero-initial-state SDOF oscillators.

    Input is linearly interpolated and zero after final sample; supply a trace
    ending at zero. Residual response included for the stated decay constants.
    """
    t = _times(times_s)
    a = _a(acceleration_m_s2, "acceleration_m_s2", 1)
    freq = _a(natural_frequencies_hz, "natural_frequencies_hz", 1)
    zeta = _pos(damping_ratio, "damping_ratio")
    if not len(freq) or np.any(freq <= 0) or a.shape != t.shape or zeta >= 1:
        raise ValueError(
            "positive frequencies, matching trace, and 0<damping<1 required"
        )
    if abs(a[-1]) > max(1e-12, np.max(np.abs(a)) * 1e-8):
        raise ValueError("shock trace must end at zero before residual ringdown")
    decay = _pos(residual_decay_constants, "residual_decay_constants")
    maxima = []
    minima = []
    relmax = []
    for fn in freq:
        wn = 2 * np.pi * fn
        dt = min(float(np.min(np.diff(t))), 1 / (40 * fn))
        end = t[-1] - t[0] + decay / (zeta * wn)
        sample_count = int(np.ceil(end / dt)) + 1
        if sample_count > 2000000:
            raise ValueError(
                "SRS grid exceeds 2 million samples; shorten record or reduce requested frequency span"
            )
        grid = np.linspace(0, end, sample_count)
        forcing = np.interp(grid, t - t[0], a, left=0, right=0)
        system = StateSpace(
            [[0, 1], [-wn * wn, -2 * zeta * wn]],
            [[0], [-1]],
            [[-wn * wn, -2 * zeta * wn], [1, 0]],
            [[0], [0]],
        )
        _, out, _ = lsim(system, forcing, grid)
        maxima.append(float(out[:, 0].max()))
        minima.append(float(out[:, 0].min()))
        relmax.append(float(np.max(np.abs(out[:, 1]))))
    return {
        "frequency_hz": freq.tolist(),
        "positive_acceleration_m_s2": maxima,
        "negative_acceleration_m_s2": minima,
        "absolute_acceleration_m_s2": np.maximum(maxima, -np.array(minima)).tolist(),
        "relative_displacement_m": relmax,
        "input_samples_per_highest_mode_period": float(
            1 / (np.max(np.diff(t)) * freq.max())
        ),
        "residual_decay_constants": decay,
        "scope": "Sampled SDOF acceleration SRS including residual ringdown; input sampling/bandwidth remains caller evidence.",
    }


def fatigue_damage(
    *,
    stress_history_pa,
    sn_amplitude_pa,
    sn_cycles,
    damage_limit,
    mean_stress_ultimate_pa=None,
):
    """Rainflow closed cycles plus residual half cycles, log-log S-N Miner damage.

    Table is stress AMPLITUDE vs cycles to failure. No extrapolation or automatic
    endurance limit; optional Goodman correction ignores beneficial negative mean.
    """
    history = _a(stress_history_pa, "stress_history_pa", 1)
    amplitudes = _a(sn_amplitude_pa, "sn_amplitude_pa", 1)
    cycles = _a(sn_cycles, "sn_cycles", 1)
    if (
        len(history) < 2
        or len(amplitudes) < 2
        or cycles.shape != amplitudes.shape
        or np.any(amplitudes <= 0)
        or np.any(cycles <= 0)
        or np.any(np.diff(amplitudes) <= 0)
        or np.any(np.diff(cycles) >= 0)
    ):
        raise ValueError(
            "increasing amplitude/decreasing life S-N table and stress history required"
        )
    unique = [float(history[0])]
    for x in history[1:]:
        if x != unique[-1]:
            unique.append(float(x))
    turning = (
        unique[:1]
        + [
            unique[i]
            for i in range(1, len(unique) - 1)
            if (unique[i] - unique[i - 1]) * (unique[i + 1] - unique[i]) < 0
        ]
        + unique[-1:]
    )
    stack = []
    counted = []
    for point in turning:
        stack.append(point)
        while len(stack) >= 3:
            earlier = abs(stack[-2] - stack[-3])
            later = abs(stack[-1] - stack[-2])
            if earlier > later:
                break
            if len(stack) == 3:
                counted.append((earlier, (stack[-3] + stack[-2]) / 2, 0.5))
                stack.pop(0)
            else:
                counted.append((earlier, (stack[-3] + stack[-2]) / 2, 1.0))
                del stack[-3:-1]
    counted.extend((abs(a - b), (a + b) / 2, 0.5) for a, b in zip(stack, stack[1:]))
    rows = []
    damage = 0.0
    ultimate = (
        None
        if mean_stress_ultimate_pa is None
        else _pos(mean_stress_ultimate_pa, "mean_stress_ultimate_pa")
    )
    for stress_range, mean, count in counted:
        if stress_range == 0:
            continue
        corrected = stress_range / 2
        if ultimate is not None:
            if mean >= ultimate:
                raise ValueError("mean stress exceeds Goodman ultimate strength")
            corrected /= 1 - max(mean, 0) / ultimate
        if not amplitudes[0] <= corrected <= amplitudes[-1]:
            raise ValueError(
                "cycle amplitude outside supplied S-N table; no extrapolation"
            )
        life = float(
            np.exp(np.interp(np.log(corrected), np.log(amplitudes), np.log(cycles)))
        )
        d = count / life
        damage += d
        rows.append(
            {
                "range_pa": stress_range,
                "mean_pa": mean,
                "count": count,
                "corrected_amplitude_pa": corrected,
                "cycles_to_failure": life,
                "damage": d,
            }
        )
    limit = _pos(damage_limit, "damage_limit")
    return {
        "cycles": rows,
        "miner_damage": damage,
        "damage_limit": limit,
        "pass": damage <= limit,
        "scope": "Rainflow/Miner uniaxial elastic screen; supplied S-N curve/environment/mean-stress model must be validated.",
    }


def fracture_margin(
    *,
    tensile_stress_pa,
    flaw_depth_m,
    geometry_factor,
    toughness_pa_sqrt_m,
    load_factor,
):
    stress = _pos(tensile_stress_pa, "tensile_stress_pa", True)
    ki = (
        _pos(geometry_factor, "geometry_factor")
        * stress
        * math.sqrt(math.pi * _pos(flaw_depth_m, "flaw_depth_m"))
        * _factor(load_factor)
    )
    return {
        "mode_i_stress_intensity_pa_sqrt_m": ki,
        "fracture": _margin(ki, toughness_pa_sqrt_m),
        "scope": "Linear elastic mode-I fracture; supplied geometry factor and valid toughness/thickness/environment required.",
    }


def temperature_allowable(*, temperature_k, table_temperature_k, table_allowable_pa):
    t = _a(table_temperature_k, "table_temperature_k", 1)
    strength = _a(table_allowable_pa, "table_allowable_pa", 1)
    query = _a(temperature_k, "temperature_k")
    if (
        len(t) < 2
        or t.shape != strength.shape
        or np.any(np.diff(t) <= 0)
        or np.any(strength <= 0)
        or np.any(query < t[0])
        or np.any(query > t[-1])
    ):
        raise ValueError(
            "positive allowable table and query inside temperature domain required"
        )
    return {
        "temperature_k": query.tolist(),
        "allowable_pa": np.interp(query, t, strength).tolist(),
        "extrapolated": False,
    }


def thermal_strain(
    *,
    alpha_per_k,
    temperature_change_k,
    length_m,
    youngs_modulus_pa,
    area_m2,
    restraint_fraction,
):
    alpha = float(alpha_per_k)
    dt = float(temperature_change_k)
    rest = float(restraint_fraction)
    if not np.isfinite([alpha, dt, rest]).all() or not 0 <= rest <= 1:
        raise ValueError("finite inputs and restraint fraction [0,1] required")
    strain = alpha * dt
    stress = -_pos(youngs_modulus_pa, "youngs_modulus_pa") * rest * strain
    return {
        "free_thermal_strain": strain,
        "free_expansion_m": strain * _pos(length_m, "length_m"),
        "restrained_stress_pa": stress,
        "restraint_force_n": stress * _pos(area_m2, "area_m2"),
        "scope": "Uniform-temperature uniaxial linear isotropic restraint model.",
    }


def thermal_network(
    *,
    times_s,
    heat_capacity_j_k,
    conductance_w_k,
    initial_temperature_k,
    heat_input_w,
    ambient_temperature_k,
    ambient_conductance_w_k,
    emissivity_area_m2,
    temperature_min_k=None,
    temperature_max_k=None,
    rtol=1e-8,
):
    """C dT/dt=sum Gij(Tj-Ti)+Q+H(Tamb-T)+eps*A*sigma(Tamb^4-T^4).

    Time-varying heat and ambient histories interpolate linearly. Capacities,
    conductances, and properties are constant. Integral state audits energy.
    """
    t = _times(times_s)
    capacity = _a(heat_capacity_j_k, "heat_capacity_j_k", 1)
    n = len(capacity)
    if not n or np.any(capacity <= 0):
        raise ValueError("positive node heat capacities required")
    g = _matrix(conductance_w_k, "conductance_w_k", n)
    if np.any(g < 0) or np.any(np.diag(g) != 0):
        raise ValueError(
            "conductance must have nonnegative off-diagonals and zero diagonal"
        )
    initial = _a(initial_temperature_k, "initial_temperature_k", 1)
    h = _a(ambient_conductance_w_k, "ambient_conductance_w_k", 1)
    epsarea = _a(emissivity_area_m2, "emissivity_area_m2", 1)
    if (
        initial.shape != (n,)
        or h.shape != (n,)
        or epsarea.shape != (n,)
        or np.any(initial <= 0)
        or np.any(h < 0)
        or np.any(epsarea < 0)
    ):
        raise ValueError("invalid thermal vector inputs")
    q = _history(heat_input_w, t, n, "heat_input_w")
    ambient = _history(ambient_temperature_k, t, n, "ambient_temperature_k")
    if np.any(ambient <= 0):
        raise ValueError("ambient temperature must be Kelvin >0")

    def rhs(time, state):
        temp = state[:n]
        if np.any(temp <= 0):
            raise ValueError("thermal model crossed absolute zero")
        env = _interp(t, ambient, time)
        ext = (
            _interp(t, q, time)
            + h * (env - temp)
            + epsarea * 5.670374419e-8 * (env**4 - temp**4)
        )
        internal = g @ temp - g.sum(axis=1) * temp
        return np.r_[(internal + ext) / capacity, ext.sum()]

    sol = solve_ivp(
        rhs,
        (t[0], t[-1]),
        np.r_[initial, 0.0],
        t_eval=t,
        rtol=_pos(rtol, "rtol"),
        atol=rtol * 1e-3,
        max_step=float(np.min(np.diff(t))) / 2,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    temp = sol.y[:n].T
    out = {
        "times_s": t.tolist(),
        "temperature_k": temp.tolist(),
        "minimum_temperature_k": temp.min(axis=0).tolist(),
        "maximum_temperature_k": temp.max(axis=0).tolist(),
        "net_external_heat_j": float(sol.y[-1, -1]),
        "energy_balance_error_j": float(
            (temp[-1] - initial) @ capacity - sol.y[-1, -1]
        ),
    }
    if (temperature_min_k is None) != (temperature_max_k is None):
        raise ValueError("supply both minimum and maximum temperature limits")
    if temperature_min_k is not None:
        low = np.broadcast_to(_a(temperature_min_k, "temperature_min_k"), (n,))
        high = np.broadcast_to(_a(temperature_max_k, "temperature_max_k"), (n,))
        if np.any(high < low):
            raise ValueError("temperature bounds reversed")
        out["temperature_limits_pass"] = bool(
            np.all(temp >= low) and np.all(temp <= high)
        )
    out["scope"] = (
        "Lumped constant-property thermal network; no boiling, phase change, ablation or generated aerothermal correlation. Heat histories must be supplied."
    )
    return out


def multilayer_conduction(
    *,
    times_s,
    thickness_m,
    conductivity_w_m_k,
    density_kg_m3,
    specific_heat_j_kg_k,
    area_m2,
    initial_temperature_k,
    left_temperature_k,
    right_temperature_k,
    left_h_w_m2_k,
    right_h_w_m2_k,
    heat_input_w,
    rtol=1e-8,
):
    """One finite-volume node per layer; half-layer series resistances at interfaces.

    Discretize thick layers into additional sublayers and demonstrate convergence.
    Boundary h=0 is insulated; positive finite h includes half-layer resistance.
    """
    t = _times(times_s)
    d = _a(thickness_m, "thickness_m", 1)
    n = len(d)
    kk = _a(conductivity_w_m_k, "conductivity_w_m_k", 1)
    rho = _a(density_kg_m3, "density_kg_m3", 1)
    cp = _a(specific_heat_j_kg_k, "specific_heat_j_kg_k", 1)
    if not n or any(v.shape != (n,) or np.any(v <= 0) for v in (d, kk, rho, cp)):
        raise ValueError("positive matching layer properties required")
    area = _pos(area_m2, "area_m2")
    capacity = rho * cp * d * area
    g = np.zeros((n, n))
    for j in range(n - 1):
        g[j, j + 1] = g[j + 1, j] = area / (
            d[j] / (2 * kk[j]) + d[j + 1] / (2 * kk[j + 1])
        )
    hl = _pos(left_h_w_m2_k, "left_h_w_m2_k", True)
    hr = _pos(right_h_w_m2_k, "right_h_w_m2_k", True)
    gl = 0 if hl == 0 else area / (1 / hl + d[0] / (2 * kk[0]))
    gr = 0 if hr == 0 else area / (1 / hr + d[-1] / (2 * kk[-1]))
    left = _history(left_temperature_k, t, 1, "left_temperature_k")[:, 0]
    right = _history(right_temperature_k, t, 1, "right_temperature_k")[:, 0]
    h = np.zeros(n)
    h[0] += gl
    h[-1] += gr
    env = np.full((len(t), n), 300.0)
    if n == 1:
        if gl + gr:
            env[:, 0] = (gl * left + gr * right) / (gl + gr)
    else:
        env[:, 0] = left
        env[:, -1] = right
    return thermal_network(
        times_s=t,
        heat_capacity_j_k=capacity,
        conductance_w_k=g,
        initial_temperature_k=initial_temperature_k,
        heat_input_w=heat_input_w,
        ambient_temperature_k=env,
        ambient_conductance_w_k=h,
        emissivity_area_m2=np.zeros(n),
        rtol=rtol,
    )


def vent_pressure(
    *,
    times_s,
    volume_m3,
    initial_pressure_pa,
    initial_temperature_k,
    ambient_pressure_pa,
    ambient_temperature_k,
    vent_area_m2,
    discharge_coefficient,
    gas_constant_j_kg_k,
    heat_capacity_ratio,
    heat_input_w,
    rtol=1e-8,
):
    """Rigid perfectly mixed ideal-gas volume with bidirectional compressible orifice.

    Choked/subsonic isentropic discharge and reservoir energy: dU/dt=Q+hin*min-
    hout*mout. No prescribed constant temperature and no artificial pressure floor.
    """
    t = _times(times_s)
    volume = _pos(volume_m3, "volume_m3")
    rg = _pos(gas_constant_j_kg_k, "gas_constant_j_kg_k")
    gamma = _pos(heat_capacity_ratio, "heat_capacity_ratio")
    if gamma <= 1:
        raise ValueError("heat_capacity_ratio must exceed one")
    area = _pos(vent_area_m2, "vent_area_m2", True)
    cd = _pos(discharge_coefficient, "discharge_coefficient")
    if cd > 1:
        raise ValueError("discharge coefficient must not exceed one")
    p0 = _pos(initial_pressure_pa, "initial_pressure_pa")
    temp0 = _pos(initial_temperature_k, "initial_temperature_k")
    pa = _history(ambient_pressure_pa, t, 1, "ambient_pressure_pa")[:, 0]
    ta = _history(ambient_temperature_k, t, 1, "ambient_temperature_k")[:, 0]
    q = _history(heat_input_w, t, 1, "heat_input_w")[:, 0]
    if np.any(pa < 0) or np.any(ta <= 0):
        raise ValueError("ambient pressure >=0 and temperature >0 required")
    cv = rg / (gamma - 1)
    cp = cv + rg
    m0 = p0 * volume / (rg * temp0)
    critical = (2 / (gamma + 1)) ** (gamma / (gamma - 1))

    def flow(pup, pdown, tup):
        if pup <= pdown or area == 0:
            return 0.0
        ratio = max(0.0, pdown / pup)
        if ratio <= critical:
            flux = (
                pup
                / math.sqrt(rg * tup)
                * math.sqrt(gamma)
                * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
            )
        else:
            flux = (
                pup
                / math.sqrt(rg * tup)
                * math.sqrt(
                    max(
                        0.0,
                        2
                        * gamma
                        / (gamma - 1)
                        * (ratio ** (2 / gamma) - ratio ** ((gamma + 1) / gamma)),
                    )
                )
            )
        return cd * area * flux

    def rhs(time, state):
        mass, energy = state[:2]
        if mass <= 0 or energy <= 0:
            raise ValueError("vent model lost positive gas mass/internal energy")
        temp = energy / (mass * cv)
        pressure = mass * rg * temp / volume
        ambientp = float(np.interp(time, t, pa))
        ambientt = float(np.interp(time, t, ta))
        heat = float(np.interp(time, t, q))
        outward = pressure >= ambientp
        mdot = (
            flow(pressure, ambientp, temp)
            if outward
            else -flow(ambientp, pressure, ambientt)
        )
        enthalpy = cp * (temp if outward else ambientt)
        return [-mdot, heat - mdot * enthalpy, mdot, heat - mdot * enthalpy]

    initial = [m0, m0 * cv * temp0, 0, 0]
    sol = solve_ivp(
        rhs,
        (t[0], t[-1]),
        initial,
        t_eval=t,
        rtol=_pos(rtol, "rtol"),
        atol=rtol * 1e-3,
        max_step=float(np.min(np.diff(t))) / 2,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    mass = sol.y[0]
    temp = sol.y[1] / (mass * cv)
    pressure = mass * rg * temp / volume
    return {
        "times_s": t.tolist(),
        "mass_kg": mass.tolist(),
        "temperature_k": temp.tolist(),
        "pressure_pa": pressure.tolist(),
        "differential_pressure_pa": (pressure - pa).tolist(),
        "mass_balance_error_kg": float(mass[-1] - m0 + sol.y[2, -1]),
        "energy_balance_error_j": float(sol.y[1, -1] - initial[1] - sol.y[3, -1]),
        "scope": "Fixed-volume ideal-gas well-mixed reservoir; supplied vent coefficient and heat history. Not a two-phase propellant model.",
    }


def deformation_clearance(
    *, nominal_gap_m, displacement_toward_gap_m, tolerance_stack_m, required_clearance_m
):
    """Gap-normal displacement/history including both bodies; worst-case additive tolerances."""
    gap = _pos(nominal_gap_m, "nominal_gap_m", True)
    required = _pos(required_clearance_m, "required_clearance_m", True)
    disp = _a(displacement_toward_gap_m, "displacement_toward_gap_m")
    tol = _a(tolerance_stack_m, "tolerance_stack_m", 1)
    if disp.size == 0 or np.any(tol < 0):
        raise ValueError(
            "nonempty displacement and nonnegative worst-case tolerances required"
        )
    minimum = gap - float(np.max(disp)) - float(tol.sum())
    return {
        "minimum_clearance_m": minimum,
        "required_clearance_m": required,
        "clearance_margin_m": minimum - required,
        "pass": minimum >= required,
        "scope": "Supplied gap-normal envelope; full CAD collision and swept geometry remain caller evidence.",
    }


def load_resultant(
    *, forces_n, application_points_m, free_moments_nm, reference_point_m
):
    """Correlated simultaneous force case; vectors share a common Cartesian frame."""
    forces = _a(forces_n, "forces_n", 2)
    points = _a(application_points_m, "application_points_m", 2)
    moments = _a(free_moments_nm, "free_moments_nm", 2)
    reference = _a(reference_point_m, "reference_point_m", 1)
    if (
        forces.shape != points.shape
        or forces.shape[1] != 3
        or moments.shape[1] != 3
        or reference.shape != (3,)
    ):
        raise ValueError(
            "forces/points require matching Nx3; moments Mx3; reference three components"
        )
    result = forces.sum(axis=0)
    moment = np.cross(points - reference, forces).sum(axis=0) + moments.sum(axis=0)
    return {
        "resultant_force_n": result.tolist(),
        "resultant_moment_nm": moment.tolist(),
        "scope": "One simultaneous load case; componentwise peaks from different times must not be combined without an explicit load-case basis.",
    }


def support_reactions(
    *,
    applied_forces_n,
    application_points_m,
    applied_moments_nm,
    support_points_m,
    support_directions,
    support_stiffness_n_m,
    reference_point_m,
):
    """Rigid body's scalar directional spring supports, linear minimum strain energy.

    Every support is a supplied axial spring; positive reaction acts in its unit
    direction. Cannot represent unilateral contact/uplift or flexible load paths.
    """
    result = load_resultant(
        forces_n=applied_forces_n,
        application_points_m=application_points_m,
        free_moments_nm=applied_moments_nm,
        reference_point_m=reference_point_m,
    )
    points = _a(support_points_m, "support_points_m", 2)
    directions = _a(support_directions, "support_directions", 2)
    stiffness = _a(support_stiffness_n_m, "support_stiffness_n_m", 1)
    ref = _a(reference_point_m, "reference_point_m", 1)
    if (
        points.shape != directions.shape
        or points.shape != (len(stiffness), 3)
        or np.any(stiffness <= 0)
    ):
        raise ValueError(
            "matching support points/directions and positive stiffness required"
        )
    if not np.allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-10):
        raise ValueError("support directions must be unit vectors")
    a = np.vstack([directions.T, np.cross(points - ref, directions).T])
    wrench = np.r_[result["resultant_force_n"], result["resultant_moment_nm"]]
    rigid = np.linalg.pinv((a * stiffness) @ a.T, rcond=1e-12) @ (-wrench)
    reactions = stiffness * (a.T @ rigid)
    residual = a @ reactions + wrench
    if np.linalg.norm(residual) > 1e-8 * max(1, np.linalg.norm(wrench)):
        raise ValueError("support system cannot equilibrate applied force/moment")
    return {
        "support_scalar_reactions_n": reactions.tolist(),
        "support_force_vectors_n": (reactions[:, None] * directions).tolist(),
        "rigid_displacement_rotation": rigid.tolist(),
        "equilibrium_residual_force_moment": residual.tolist(),
        "scope": "Rigid body on supplied bilateral directional springs; stiffness determines indeterminate reaction sharing.",
    }


def inertial_loads(
    *,
    masses_kg,
    positions_m,
    reference_point_m,
    reference_acceleration_m_s2,
    angular_velocity_rad_s,
    angular_acceleration_rad_s2,
):
    """D'Alembert forces at discrete rigidly attached masses; gravity is a separate applied load."""
    mass = _a(masses_kg, "masses_kg", 1)
    points = _a(positions_m, "positions_m", 2)
    ref = _a(reference_point_m, "reference_point_m", 1)
    accel = _a(reference_acceleration_m_s2, "reference_acceleration_m_s2", 1)
    omega = _a(angular_velocity_rad_s, "angular_velocity_rad_s", 1)
    alpha = _a(angular_acceleration_rad_s2, "angular_acceleration_rad_s2", 1)
    if (
        points.shape != (len(mass), 3)
        or np.any(mass < 0)
        or any(v.shape != (3,) for v in [ref, accel, omega, alpha])
    ):
        raise ValueError("invalid mass locations or three-component motion vectors")
    r = points - ref
    local_accel = accel + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
    force = -mass[:, None] * local_accel
    out = load_resultant(
        forces_n=force,
        application_points_m=points,
        free_moments_nm=np.zeros((0, 3)),
        reference_point_m=ref,
    )
    out.update(
        inertial_forces_n=force.tolist(),
        local_acceleration_m_s2=local_accel.tolist(),
        total_mass_kg=float(mass.sum()),
    )
    return out


def surface_loads(
    *,
    patch_pressure_pa,
    patch_areas_m2,
    outward_normals,
    patch_centroids_m,
    reference_point_m,
):
    """Pressure positive into a surface with supplied outward unit normal."""
    pressure = _a(patch_pressure_pa, "patch_pressure_pa", 1)
    area = _a(patch_areas_m2, "patch_areas_m2", 1)
    normals = _a(outward_normals, "outward_normals", 2)
    if (
        pressure.shape != area.shape
        or normals.shape != (len(area), 3)
        or np.any(area <= 0)
        or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-10)
    ):
        raise ValueError("matching pressure/area and unit outward normals required")
    force = -pressure[:, None] * area[:, None] * normals
    out = load_resultant(
        forces_n=force,
        application_points_m=patch_centroids_m,
        free_moments_nm=np.zeros((0, 3)),
        reference_point_m=reference_point_m,
    )
    out["patch_forces_n"] = force.tolist()
    return out


def combine_load_cases(*, case_forces_n, case_moments_nm, combination_coefficients):
    """Linear combinations of explicitly correlated source cases; no statistical independence assumed."""
    force = _a(case_forces_n, "case_forces_n", 2)
    moment = _a(case_moments_nm, "case_moments_nm", 2)
    coeff = _a(combination_coefficients, "combination_coefficients", 2)
    if (
        force.shape != moment.shape
        or force.shape[1] != 3
        or coeff.shape[1] != len(force)
    ):
        raise ValueError("load case/combination shape mismatch")
    return {
        "combined_force_n": (coeff @ force).tolist(),
        "combined_moment_nm": (coeff @ moment).tolist(),
    }


def thermoelastic_static(
    *,
    stiffness_n_m,
    mechanical_force_n,
    constrained_dofs,
    temperature_k,
    stress_free_temperature_k,
    thermal_force_n_k,
    stress_recovery_pa_m,
    thermal_stress_recovery_pa_k,
    prescribed_displacement_m=None,
):
    """Supplied FE equivalent thermal-load/recovery matrices include constitutive and expansion data.

    sigma=B_u*u+B_T*(T-T0); B_T normally includes negative free thermal strain.
    Sign/shape/mesh/material calibration belongs to the supplied FE model.
    """
    k = _matrix(stiffness_n_m, "stiffness_n_m")
    n = len(k)
    temp = _a(temperature_k, "temperature_k", 1)
    base = _a(stress_free_temperature_k, "stress_free_temperature_k", 1)
    ft = _a(thermal_force_n_k, "thermal_force_n_k", 2)
    bt = _a(thermal_stress_recovery_pa_k, "thermal_stress_recovery_pa_k", 2)
    bu = _a(stress_recovery_pa_m, "stress_recovery_pa_m", 2)
    mechanical = _a(mechanical_force_n, "mechanical_force_n", 1)
    if (
        temp.shape != base.shape
        or ft.shape != (n, len(temp))
        or bt.shape != (bu.shape[0], len(temp))
        or mechanical.shape != (n,)
    ):
        raise ValueError("thermal FE matrix dimensions do not match")
    delta = temp - base
    out = fe_static(
        stiffness_n_m=k,
        force_n=mechanical + ft @ delta,
        constrained_dofs=constrained_dofs,
        prescribed_displacement_m=prescribed_displacement_m,
        stress_recovery_pa_m=bu,
        stress_offset_pa=bt @ delta,
    )
    out.update(
        equivalent_thermal_force_n=(ft @ delta).tolist(),
        temperature_change_k=delta.tolist(),
        scope="Linear supplied-matrix thermoelastic FEA; temperature-dependent matrices require externally supplied appropriate material state.",
    )
    return out


def coupled_load_modal(
    *,
    times_s,
    stiffness_n_m,
    mass_kg,
    damping_ns_m,
    force_n,
    initial_displacement_m,
    initial_velocity_m_s,
    base_acceleration_m_s2=None,
    base_influence=None,
    rtol=1e-8,
):
    transient = fe_transient(
        times_s=times_s,
        stiffness_n_m=stiffness_n_m,
        mass_kg=mass_kg,
        damping_ns_m=damping_ns_m,
        force_n=force_n,
        initial_displacement_m=initial_displacement_m,
        initial_velocity_m_s=initial_velocity_m_s,
        base_acceleration_m_s2=base_acceleration_m_s2,
        base_influence=base_influence,
        rtol=rtol,
    )
    return {
        "transient": transient,
        "modal": modal(stiffness_n_m=stiffness_n_m, mass_kg=mass_kg),
    }


def section_joint_strength(*, section_inputs, joints):
    """Composite strength screen; joint loads must come from an evidenced load distribution."""
    section = section_stress(**section_inputs)
    joint_results = [joint_strength(**j) for j in joints]
    if not joint_results:
        raise ValueError("at least one joint input is required")
    passed = (
        section["yield"]["pass"]
        and section["ultimate_equivalent_stress_screen"]["pass"]
        and all(
            j["bolt_pass"]
            and j["no_joint_separation"]
            and j["bearing"]["pass"]
            and j["weld"]["pass"]
            for j in joint_results
        )
    )
    return {
        "section": section,
        "joints": joint_results,
        "all_screens_pass": bool(passed),
    }


def temperature_strength(
    *,
    temperature_k,
    stress_demand_pa,
    table_temperature_k,
    table_allowable_pa,
    load_factor,
):
    allow = temperature_allowable(
        temperature_k=temperature_k,
        table_temperature_k=table_temperature_k,
        table_allowable_pa=table_allowable_pa,
    )
    allowable = np.asarray(allow["allowable_pa"])
    stress = _a(stress_demand_pa, "stress_demand_pa")
    stress, allowable = np.broadcast_arrays(stress, allowable)
    if np.any(stress < 0):
        raise ValueError("equivalent stress demand must be nonnegative")
    utilization = stress * _factor(load_factor) / allowable
    return {
        "temperature_k": np.asarray(temperature_k).tolist(),
        "allowable_pa": allowable.tolist(),
        "utilization": utilization.tolist(),
        "maximum_utilization": float(utilization.max()),
        "pass": bool(np.all(utilization <= 1)),
    }


def corrosion_loss(
    *,
    initial_thickness_m,
    corrosion_rate_m_s,
    oxidation_loss_m,
    exposure_time_s,
    required_remaining_thickness_m,
):
    """Uniform linear recession screen; supplied rates must be exposure/material specific."""
    initial = _pos(initial_thickness_m, "initial_thickness_m")
    rate = _pos(corrosion_rate_m_s, "corrosion_rate_m_s", True)
    oxidation = _pos(oxidation_loss_m, "oxidation_loss_m", True)
    duration = _pos(exposure_time_s, "exposure_time_s", True)
    required = _pos(required_remaining_thickness_m, "required_remaining_thickness_m")
    remaining = initial - rate * duration - oxidation
    return {
        "remaining_thickness_m": remaining,
        "loss_m": initial - remaining,
        "thickness_margin_m": remaining - required,
        "pass": remaining >= required,
        "scope": "Uniform supplied-rate loss; not a pitting, galvanic, oxidation-kinetics, or oxygen-compatibility prediction.",
    }


def impulse_momentum(
    *, mass_kg, inertia_kg_m2, linear_impulse_ns, angular_impulse_nms, impulse_offset_m
):
    """Instantaneous rigid-body velocity changes about the center of mass."""
    mass = _pos(mass_kg, "mass_kg")
    inertia = _matrix(inertia_kg_m2, "inertia_kg_m2", 3, True)
    impulse = _a(linear_impulse_ns, "linear_impulse_ns", 1)
    angular = _a(angular_impulse_nms, "angular_impulse_nms", 1)
    offset = _a(impulse_offset_m, "impulse_offset_m", 1)
    if any(v.shape != (3,) for v in [impulse, angular, offset]):
        raise ValueError("three-component impulse and offset vectors required")
    total = angular + np.cross(offset, impulse)
    return {
        "delta_velocity_m_s": (impulse / mass).tolist(),
        "delta_angular_velocity_rad_s": np.linalg.solve(inertia, total).tolist(),
        "total_angular_impulse_nms": total.tolist(),
        "scope": "Instantaneous impulse; finite contact/release sequence and relative propagation are separate.",
    }


def pendulum_slosh(
    *,
    times_s,
    slosh_mass_kg,
    pendulum_length_m,
    restoring_acceleration_m_s2,
    damping_ratio,
    support_acceleration_m_s2,
    initial_angle_rad,
    initial_rate_rad_s,
    rtol=1e-8,
):
    """Planar equivalent pendulum with nonlinear sin/cos and viscous damping.

    Equivalent length, participating mass and damping require tank/fill/acceleration
    calibration. This is not a CFD/free-surface prediction or zero-g capillary model.
    """
    t = _times(times_s)
    mass = _pos(slosh_mass_kg, "slosh_mass_kg")
    length = _pos(pendulum_length_m, "pendulum_length_m")
    g = _pos(restoring_acceleration_m_s2, "restoring_acceleration_m_s2")
    zeta = _pos(damping_ratio, "damping_ratio", True)
    accel = _history(support_acceleration_m_s2, t, 1, "support_acceleration_m_s2")[:, 0]
    initial = _a([initial_angle_rad, initial_rate_rad_s], "initial_state")
    wn = math.sqrt(g / length)

    def rhs(time, state):
        theta, rate = state
        forcing = float(np.interp(time, t, accel))
        return [
            rate,
            -2 * zeta * wn * rate
            - g / length * math.sin(theta)
            - forcing / length * math.cos(theta),
        ]

    sol = solve_ivp(
        rhs,
        (t[0], t[-1]),
        initial,
        t_eval=t,
        rtol=_pos(rtol, "rtol"),
        atol=rtol * 1e-3,
        max_step=min(float(np.min(np.diff(t))) / 2, 1 / (20 * wn)),
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    theta, rate = sol.y
    alpha = np.array([rhs(time, state)[1] for time, state in zip(t, sol.y.T)])
    # Net horizontal force on the support is minus mass times absolute bob acceleration.
    bob_ax = accel + length * (alpha * np.cos(theta) - rate**2 * np.sin(theta))
    return {
        "times_s": t.tolist(),
        "angle_rad": theta.tolist(),
        "angular_rate_rad_s": rate.tolist(),
        "lateral_slosh_displacement_m": (length * np.sin(theta)).tolist(),
        "force_on_support_x_n": (-mass * bob_ax).tolist(),
        "small_angle_frequency_hz": wn / (2 * np.pi),
        "scope": "Calibrated equivalent planar pendulum; damping reaction is included in net support force via bob acceleration.",
    }


def pressure_psd_response(
    *,
    frequency_hz,
    exterior_pressure_psd_pa2_hz,
    pressure_transmission_amplitude,
    load_area_vector_m2,
    stiffness_n_m,
    mass_kg,
    damping_ns_m,
):
    """Fully coherent pressure input through SUPPLIED acoustic transmission and FE force distribution.

    Transmission amplitudes and load-area vector are external acoustic/geometry
    models. This computes the mechanical response; it does not predict transmission.
    """
    freq = _a(frequency_hz, "frequency_hz", 1)
    psd = _a(exterior_pressure_psd_pa2_hz, "exterior_pressure_psd_pa2_hz", 1)
    trans = _a(pressure_transmission_amplitude, "pressure_transmission_amplitude", 1)
    k = _matrix(stiffness_n_m, "stiffness_n_m", positive=True)
    n = len(k)
    m = _matrix(mass_kg, "mass_kg", n, True)
    c = _matrix(damping_ns_m, "damping_ns_m", n)
    area = _a(load_area_vector_m2, "load_area_vector_m2", 1)
    if (
        len(freq) < 2
        or np.any(np.diff(freq) <= 0)
        or freq[0] < 0
        or psd.shape != freq.shape
        or trans.shape != freq.shape
        or np.any(psd < 0)
        or np.any(trans < 0)
        or area.shape != (n,)
    ):
        raise ValueError(
            "invalid pressure spectrum, transmission, or area distribution"
        )
    if np.min(eigvalsh(c)) < 0:
        raise ValueError("nonnegative damping required")
    out = []
    for f in freq:
        w = 2 * np.pi * f
        response = -w * w * np.linalg.solve(k - w * w * m + 1j * w * c, area)
        out.append(np.abs(response) ** 2)
    response_psd = np.asarray(out) * (psd * trans * trans)[:, None]
    return {
        "frequency_hz": freq.tolist(),
        "acceleration_psd_m2_s4_hz": response_psd.tolist(),
        "acceleration_rms_m_s2": np.sqrt(
            trapezoid(response_psd, freq, axis=0)
        ).tolist(),
        "interior_pressure_rms_pa": float(
            np.sqrt(trapezoid(psd * trans * trans, freq))
        ),
        "scope": "Fully coherent supplied pressure field/transmission; no independent acoustic transmission prediction.",
    }


def inertia_and_stress(
    *,
    inertial_inputs,
    applied_forces_n,
    application_points_m,
    applied_moments_nm,
    section_properties,
):
    """Section-cut D'Alembert balance, with all distal masses/applied forces supplied.

    All coordinates and kinematics must share the section frame (x is axial).
    Caller must provide complete loads, including gravity and interface reactions.
    """
    inertial = inertial_loads(**inertial_inputs)
    applied = load_resultant(
        forces_n=applied_forces_n,
        application_points_m=application_points_m,
        free_moments_nm=applied_moments_nm,
        reference_point_m=inertial_inputs["reference_point_m"],
    )
    force = -(np.array(inertial["resultant_force_n"]) + applied["resultant_force_n"])
    moment = -(
        np.array(inertial["resultant_moment_nm"]) + applied["resultant_moment_nm"]
    )
    stress = section_stress(
        axial_force_n=force[0],
        shear_y_n=force[1],
        shear_z_n=force[2],
        torque_nm=moment[0],
        bending_y_nm=moment[1],
        bending_z_nm=moment[2],
        **section_properties,
    )
    return {
        "inertial_loads": inertial,
        "cut_force_n": force.tolist(),
        "cut_moment_nm": moment.tolist(),
        "section": stress,
    }


def strength_fracture(
    *,
    temperature_k,
    equivalent_stress_pa,
    tensile_stress_pa,
    table_temperature_k,
    table_allowable_pa,
    table_toughness_pa_sqrt_m,
    flaw_depth_m,
    geometry_factor,
    load_factor,
):
    strength = temperature_strength(
        temperature_k=temperature_k,
        stress_demand_pa=equivalent_stress_pa,
        table_temperature_k=table_temperature_k,
        table_allowable_pa=table_allowable_pa,
        load_factor=load_factor,
    )
    toughness = temperature_allowable(
        temperature_k=temperature_k,
        table_temperature_k=table_temperature_k,
        table_allowable_pa=table_toughness_pa_sqrt_m,
    )["allowable_pa"]
    if np.ndim(toughness):
        raise ValueError("strength_fracture requires one material temperature per case")
    fracture = fracture_margin(
        tensile_stress_pa=tensile_stress_pa,
        flaw_depth_m=flaw_depth_m,
        geometry_factor=geometry_factor,
        toughness_pa_sqrt_m=toughness,
        load_factor=load_factor,
    )
    return {
        "strength": strength,
        "fracture": fracture,
        "all_screens_pass": bool(strength["pass"] and fracture["fracture"]["pass"]),
    }


def degraded_strength(
    *, base_allowable_pa, retained_strength_fractions, stress_demand_pa, load_factor
):
    """Scenario sweep of supplied retained strength, no degradation-rate prediction."""
    allowable = _pos(base_allowable_pa, "base_allowable_pa")
    fractions = _a(retained_strength_fractions, "retained_strength_fractions", 1)
    if not len(fractions) or np.any(fractions <= 0) or np.any(fractions > 1):
        raise ValueError("retained fractions must lie in (0,1]")
    demand = _pos(stress_demand_pa, "stress_demand_pa", True) * _factor(load_factor)
    return {
        "scenarios": [
            dict(retained_strength_fraction=float(f), **_margin(demand, allowable * f))
            for f in fractions
        ]
    }


def burst_capacity(
    *,
    radius_m,
    thickness_m,
    material_ultimate_pa,
    weld_efficiency,
    penetration_stress_factor,
    required_pressure_pa,
    load_factor,
):
    """Thin-cylinder hoop-strength burst screen, with supplied joint efficiency and local Kt.

    Does not predict ductile fracture, crack growth or nonlinear burst behavior.
    """
    radius = _pos(radius_m, "radius_m")
    thickness = _pos(thickness_m, "thickness_m")
    efficiency = _pos(weld_efficiency, "weld_efficiency")
    kt = _pos(penetration_stress_factor, "penetration_stress_factor")
    if thickness / radius > 0.1 or efficiency > 1 or kt < 1:
        raise ValueError("t/r<=.1, efficiency<=1, and Kt>=1 required")
    pressure = (
        _pos(material_ultimate_pa, "material_ultimate_pa")
        * efficiency
        * thickness
        / (radius * kt)
    )
    return {
        "pressure_capacity_pa": pressure,
        "pressure_screen": _margin(
            _pos(required_pressure_pa, "required_pressure_pa", True)
            * _factor(load_factor),
            pressure,
        ),
        "scope": "Thin-cylinder hoop ultimate screen; qualification requires applicable nonlinear/fracture analysis and burst evidence.",
    }


FUNCTIONS = {
    "section_stress": section_stress,
    "tank_stress": tank_stress,
    "buckling": buckling,
    "buckling_eigenvalue": buckling_eigenvalue,
    "joint_strength": joint_strength,
    "fe_static": fe_static,
    "modal": modal,
    "fe_transient": fe_transient,
    "base_psd_response": base_psd_response,
    "shock_response_spectrum": shock_response_spectrum,
    "fatigue_damage": fatigue_damage,
    "fracture_margin": fracture_margin,
    "temperature_allowable": temperature_allowable,
    "thermal_strain": thermal_strain,
    "thermal_network": thermal_network,
    "multilayer_conduction": multilayer_conduction,
    "vent_pressure": vent_pressure,
    "deformation_clearance": deformation_clearance,
    "load_resultant": load_resultant,
    "support_reactions": support_reactions,
    "inertial_loads": inertial_loads,
    "surface_loads": surface_loads,
    "combine_load_cases": combine_load_cases,
    "thermoelastic_static": thermoelastic_static,
    "coupled_load_modal": coupled_load_modal,
    "section_joint_strength": section_joint_strength,
    "temperature_strength": temperature_strength,
    "corrosion_loss": corrosion_loss,
    "impulse_momentum": impulse_momentum,
    "pendulum_slosh": pendulum_slosh,
    "pressure_psd_response": pressure_psd_response,
    "inertia_and_stress": inertia_and_stress,
    "strength_fracture": strength_fracture,
    "degraded_strength": degraded_strength,
    "burst_capacity": burst_capacity,
}

# Exact workbook activity texts. Multi-physics activities may need several kernels;
# these mappings expose the applicable calculation, never assert activity completion.
KERNEL_ACTIVITIES = {
    **dict.fromkeys(
        [
            "Combined-load finite-element stress calculations.",
            "Combined-load stress finite element analysis.",
            "Combined-load tank stress finite element analysis.",
            "Asymmetric-thrust frame stress finite element analysis.",
            "Tank penetration stress finite element analysis.",
            "Propulsion-interface elastic deflection simulation.",
            "Stage-interface elastic deflection simulation.",
            "Thrust-frame elastic deflection simulation.",
        ],
        "fe_static",
    ),
    **dict.fromkeys(
        [
            "Payload coupled-load response calculations.",
            "Payload-interface coupled-load response calculations.",
            "Coupled-load axial-response calculations.",
            "Coupled-load lateral-response calculations.",
            "Coupled vehicle–payload acceleration simulation.",
            "Coupled fairing–payload deflection simulation.",
            "Separation-impulse structural transient simulation.",
            "Slosh-driven tank-support deformation simulation.",
        ],
        "fe_transient",
    ),
    **dict.fromkeys(
        [
            "Axial finite-element modal eigenvalue calculation.",
            "Lateral finite-element modal eigenvalue calculation.",
            "Payload-adapter modal eigenvalue analysis.",
            "Stage modal eigenvalue analysis.",
        ],
        "modal",
    ),
    **dict.fromkeys(
        [
            "Axial inertia and stress calculations.",
            "Lateral inertia and stress calculations.",
            "Handling-load static stress calculations.",
            "Ground-support reaction and handling-stress calculations.",
            "Ultimate-load stress and joint-strength calculations.",
            "Thrust-frame member and joint stress calculations.",
        ],
        "section_stress",
    ),
    "Combined-load tank yield stress calculations.": "tank_stress",
    "Imperfection-sensitive shell and column buckling calculations.": "buckling",
    "Imperfection-sensitive pressurized-shell buckling calculations.": "buckling",
    "Bending stress and shell buckling calculations.": "buckling",
    "Buckling eigenvalue and load-path calculations.": "buckling_eigenvalue",
    **dict.fromkeys(
        [
            "Bearing, fastener and weld stress calculations.",
            "Mount fastener and bearing stress calculations.",
            "Mount stress and fastener-preload calculations.",
            "Avionics attachment strength calculations.",
            "Tank attachment strength calculations.",
        ],
        "joint_strength",
    ),
    **dict.fromkeys(
        [
            "Vibration and acoustic spectral-response calculations.",
            "Fairing acoustic transmission and spectral-response calculations.",
            "Payload-interface modal vibration-response calculations.",
        ],
        "base_psd_response",
    ),
    **dict.fromkeys(
        [
            "Shock propagation and response-spectrum calculations.",
            "Separation shock-response spectrum calculation.",
        ],
        "shock_response_spectrum",
    ),
    **dict.fromkeys(
        [
            "Stress-cycle counting and cumulative damage calculation.",
            "Tank pressure-cycle fatigue-life calculation.",
        ],
        "fatigue_damage",
    ),
    "Flaw-based stress-intensity calculations.": "fracture_margin",
    "Cryogenic strength and fracture-margin calculations.": "strength_fracture",
    "Temperature-dependent material allowable interpolation.": "temperature_allowable",
    **dict.fromkeys(
        [
            "Thermal-strain and restraint-force calculations.",
            "Differential-expansion and joint-stress calculations.",
            "Cryogenic thermal-contraction calculations.",
        ],
        "thermal_strain",
    ),
    **dict.fromkeys(
        [
            "Equipment transient heat-balance simulation.",
            "Payload transient heat-balance simulation.",
            "Transient heat-balance simulation.",
            "Payload-compartment transient heat-balance simulation.",
            "Transient payload-compartment thermal network simulation.",
            "Ambient-temperature thermal-response calculations.",
            "Chilldown transient heat-balance calculations.",
            "Transfer-line transient heat-balance calculations.",
            "Payload-volume heat-balance and airflow calculations.",
        ],
        "thermal_network",
    ),
    "Multilayer transient heat-conduction simulation.": "multilayer_conduction",
    "Transient compressible vent-flow simulation.": "vent_pressure",
    **dict.fromkeys(
        [
            "Deformed tank-envelope clearance calculation.",
            "Deformed-envelope clearance calculation.",
            "Deformed-envelope tolerance stack-up.",
            "CAD clearance and tolerance-stack calculations.",
            "Interface tolerance-stack calculations.",
            "Harness swept-envelope clearance calculation.",
        ],
        "deformation_clearance",
    ),
    **dict.fromkeys(
        [
            "Free-body load-path calculations.",
            "Section free-body equilibrium calculations.",
            "Stage free-body load-path calculations.",
            "Independent spacecraft free-body calculations.",
            "Interstage joint load-path calculations.",
        ],
        "load_resultant",
    ),
    **dict.fromkeys(
        [
            "Avionics support-reaction calculations.",
            "Structural support-reaction calculation.",
            "Engine-reaction and support-load calculations.",
            "Equipment and propellant support-reaction calculations.",
            "Tank reaction-force calculation.",
            "Thrust-frame free-body and support-reaction calculations.",
            "Hold-down reaction-load calculations.",
            "Lift-load and rigging-force calculations.",
        ],
        "support_reactions",
    ),
    "Distributed-mass inertial and support-load calculations.": "inertial_loads",
    "Aerodynamic surface-load integration.": "surface_loads",
    "Correlated load-case combination.": "combine_load_cases",
    **dict.fromkeys(
        [
            "Cryogenic tank thermoelastic stress simulation.",
            "Interstage thermoelastic deflection simulation.",
            "Thermoelastic displacement simulation.",
            "Thermoelastic interface deflection simulation.",
            "Thrust-frame thermoelastic deflection simulation.",
        ],
        "thermoelastic_static",
    ),
    **dict.fromkeys(
        [
            "Payload coupled-load and modal response calculations.",
            "Payload-interface coupled-load and modal response calculations.",
        ],
        "coupled_load_modal",
    ),
    **dict.fromkeys(
        [
            "Ultimate-load stress and joint-strength calculations.",
            "Thrust-frame member and joint stress calculations.",
            "Interstage section-force and attachment-strength calculations.",
            "Stage section-force and joint-strength calculations.",
            "Torsional shear-flow and joint-strength calculations.",
        ],
        "section_joint_strength",
    ),
    "Temperature-dependent structural strength calculations.": "temperature_strength",
    "Corrosion-rate and oxidation-loss calculations.": "corrosion_loss",
    **dict.fromkeys(
        [
            "Release angular impulse–momentum calculation.",
            "Release impulse–momentum calculation.",
        ],
        "impulse_momentum",
    ),
    "Reduced-order propellant slosh simulation.": "pendulum_slosh",
    "Fairing acoustic transmission and spectral-response calculations.": "pressure_psd_response",
    "Transport shock/vibration structural-response calculations.": "base_psd_response",
    **dict.fromkeys(
        [
            "Axial inertia and stress calculations.",
            "Lateral inertia and stress calculations.",
        ],
        "inertia_and_stress",
    ),
    "Degraded-property structural margin sensitivity sweep.": "degraded_strength",
    "Shell, weld and penetration burst-capacity calculations.": "burst_capacity",
}
