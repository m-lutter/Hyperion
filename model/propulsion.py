"""Installed engine thrust and source tank liquid/ullage checks.

These helpers serve the assigned flight analysis and source sizing results.
Inputs are explicit SI values; screening results do not establish qualification.
"""

import math


def _positive(value, name, zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(
            f'{name} must be finite and {"nonnegative" if zero else "positive"}.'
        )
    return value


def thrust_sum(*, per_engine_thrust_N, engine_count):
    """Installed thrust in N for identical engines at the supplied operating point."""
    engine_thrust = _positive(per_engine_thrust_N, "per-engine thrust", True)
    if int(engine_count) != engine_count or engine_count < 1:
        raise ValueError("Engine count must be a positive integer.")
    return {
        "installed_thrust_N": engine_thrust * int(engine_count),
        "engine_count": int(engine_count),
    }


def liquid_ullage(*, tank_volume_m3, loaded_liquid_mass_kg, liquid_density_kg_m3):
    volume = _positive(tank_volume_m3, "tank volume")
    mass = _positive(loaded_liquid_mass_kg, "loaded liquid mass", True)
    liquid = mass / _positive(liquid_density_kg_m3, "liquid density")
    ullage = volume - liquid
    return {
        "liquid_volume_m3": liquid,
        "ullage_volume_m3": ullage,
        "ullage_fraction": ullage / volume,
        "geometric_fill_pass": bool(ullage >= 0),
    }
