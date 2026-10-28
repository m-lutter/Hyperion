"""Read the editable inputs and show exactly what a run will use.

The three TOML files under ``inputs/`` are the authoritative settings. Values
can reuse another input with ``{ ref = "vehicle.stages.1.startup_propellant_kg" }``.
Paths use dictionary keys and zero-based list positions. An optional ``scale``
multiplies a numeric reference, for example diameter to radius. No expressions
or Python code are evaluated. References are resolved before any calculation.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tomllib

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"


def _folder(path=None):
    path = Path(path) if path is not None else INPUTS
    return path if path.is_dir() else path.parent


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _documents(path=None):
    folder = _folder(path)
    documents = {}
    for name in ("vehicle", "flight", "checks"):
        with (folder / f"{name}.toml").open("rb") as stream:
            documents[name] = tomllib.load(stream)
    return documents, _read_json(folder / "requirements/mission_cases.json")


def _context(documents, mission_cases):
    context = {}
    for name in ("vehicle", "flight"):
        for key, value in documents[name].items():
            if key == "choices":
                continue
            if key in context:
                raise ValueError(f"Input section {key!r} is defined in two files")
            context[key] = value
    context["checks"] = documents["checks"]
    context["mission_cases"] = mission_cases
    return context


def _at_path(context, path):
    value = context
    try:
        for part in path.split("."):
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Unknown input reference: {path}") from error
    return value


def _resolve(value, context, chain=()):
    if isinstance(value, dict):
        if "ref" in value:
            if set(value) - {"ref", "scale"}:
                raise ValueError("An input reference may only contain ref and scale")
            path = value["ref"]
            if not isinstance(path, str):
                raise ValueError("An input reference must be a dotted text path")
            if path in chain:
                raise ValueError(f"Circular input reference: {' -> '.join((*chain, path))}")
            result = _resolve(_at_path(context, path), context, (*chain, path))
            if "scale" in value:
                if isinstance(result, bool) or not isinstance(result, (int, float)):
                    raise ValueError(f"Scaled reference {path} must be numeric")
                scale = value["scale"]
                if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                    raise ValueError(f"Scale for {path} must be numeric")
                result *= scale
            return result
        return {key: _resolve(item, context, chain) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, context, chain) for item in value]
    return copy.deepcopy(value)


def load_config(path=None):
    """Return a fresh flight dictionary from vehicle.toml and flight.toml.

    ``path`` may select an inputs directory or either TOML file in it. Explicit
    JSON files remain supported for small independent tests and old run records.
    Aerodynamic paths are relative to the repository root, as in the model.
    """
    if path is not None and Path(path).suffix.lower() == ".json":
        return _read_json(path)
    documents, mission_cases = _documents(path)
    context = _context(documents, mission_cases)
    config = {
        key: _resolve(value, context)
        for key, value in context.items()
        if key not in {"checks", "mission_cases", "departure"}
    }
    for stage in config["vehicle"]["stages"]:
        total = stage["propellant_kg"]
        tank_total = sum(tank["loaded_mass_kg"] for tank in stage["tanks"])
        if not math.isclose(total, tank_total, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError(
                f"{stage['name']}: propellant_kg ({total:g} kg) must equal the "
                f"sum of tank loaded_mass_kg ({tank_total:g} kg). Update both "
                "the stage total and tank component masses in inputs/vehicle.toml."
            )
    return config


def load_checks(path=None):
    """Return calculation settings and activity-specific acceptance rules.

    Activity entries remain separate from calculation defaults so an activity
    override cannot silently change other checks. Studies and examples are not
    submitted as verification evidence. An explicit legacy JSON file is accepted.
    """
    if path is not None and Path(path).suffix.lower() == ".json":
        return _read_json(path)
    documents, mission_cases = _documents(path)
    context = _context(documents, mission_cases)
    checks = documents["checks"]
    calculations = checks.get("calculations", {})
    activities = checks.get("activities", {})
    duplicate = calculations.keys() & activities.keys()
    if duplicate:
        raise ValueError(f"Duplicate calculation/activity names: {sorted(duplicate)}")
    return _resolve({**calculations, **activities}, context)


def load_mission_cases(path=None):
    """Read the sourced orbital state vectors and their provenance."""
    if path is not None and Path(path).suffix.lower() == ".json":
        return _read_json(path)
    return _read_json(_folder(path) / "requirements/mission_cases.json")


def load_study(name, path=None):
    """Return a named study's inputs, choices and basis from checks.toml."""
    documents, mission_cases = _documents(path)
    studies = documents["checks"].get("studies", {})
    if name not in studies:
        raise KeyError(f"Unknown study {name!r}; choose from {', '.join(studies)}")
    return _resolve(studies[name], _context(documents, mission_cases))


def _leaves(value, path=""):
    """Keep numeric schedules in one readable row; expand records into rows."""
    if isinstance(value, dict) and "ref" not in value:
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list) and any(isinstance(item, dict) for item in value):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}.{index}")
    else:
        yield path, value


def _display(value):
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return value


def _units(path):
    key = path.rsplit(".", 1)[-1]
    for suffix, units in (
        ("km2_s2", "km²/s²"), ("deg_s2", "deg/s²"), ("rad_s", "rad/s"),
        ("deg_s", "deg/s"), ("kg_m3", "kg/m³"), ("m3_s2", "m³/s²"),
        ("m_s", "m/s"), ("m2", "m²"), ("m3", "m³"), ("Nm", "N m"),
        ("nm", "N m"), ("kg", "kg"), ("deg", "deg"), ("mm", "mm"),
        ("Pa", "Pa"), ("pa", "Pa"), ("K", "K"), ("N", "N"),
        ("n", "N"), ("m", "m"), ("s", "s"), ("g", "g"),
        ("days", "days"),
    ):
        if key.endswith("_" + suffix):
            return units
    return "—"


def input_tables(path=None):
    """Return grouped tables of every editable setting, resolved before display.

    Each row names the saved key and its file. Reused values also show their
    reference, so the table identifies which definition should be edited.
    Sourced orbital vectors have their own file, outside the editable tables.
    """
    documents, mission_cases = _documents(path)
    context = _context(documents, mission_cases)
    tables = {}
    for filename, document in documents.items():
        for section, settings in document.items():
            if section == "choices":
                continue
            rows = []
            for key, raw in _leaves(settings, section):
                resolved = _resolve(raw, context)
                source = f"inputs/{filename}.toml → {key}"
                if isinstance(raw, dict) and "ref" in raw:
                    source += f" (uses {raw['ref']}"
                    source += f" × {raw['scale']}" if "scale" in raw else ""
                    source += ")"
                # Reused records are expanded too: hidden JSON is never a value.
                for subkey, value in _leaves(resolved, key):
                    rows.append({"Input": subkey, "Value": _display(value),
                                 "Units": _units(subkey), "Source": source})
            tables[f"{filename}: {section.replace('_', ' ').title()}"] = pd.DataFrame(
                rows, columns=["Input", "Value", "Units", "Source"]
            )
    return tables


def choices_table(path=None):
    """Show selected values directly from saved inputs, alongside their basis."""
    documents, mission_cases = _documents(path)
    context = _context(documents, mission_cases)
    rows = []
    for document in documents.values():
        for choice in document.get("choices", []):
            selections = []
            for path in choice["values"]:
                value = _resolve(_at_path(context, path), context)
                units = _units(path)
                suffix = f" {units}" if units != "—" else ""
                selections.append(f"{path.rsplit('.', 1)[-1]}: {_display(value)}{suffix}")
            rows.append({"Choice": choice["label"],
                         "Selection": "; ".join(selections), "Basis": choice["basis"]})
    return pd.DataFrame(rows, columns=["Choice", "Selection", "Basis"])
