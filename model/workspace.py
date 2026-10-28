"""Reuse the shared ascent only while its code, inputs and saved files match."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .inputs import load_config
from .trajectory import Simulation, run_trajectory, summarize

ROOT = Path(__file__).resolve().parents[1]


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    """Identify the actual ascent model, loader, settings and selected aero table."""
    config = load_config()
    aero_path = Path(config["aero"]["file"])
    if not aero_path.is_absolute():
        aero_path = ROOT / aero_path
    aero_path = aero_path.resolve()
    paths = [
        ROOT / "model/trajectory.py",
        ROOT / "model/inputs.py",
        ROOT / "inputs/vehicle.toml",
        ROOT / "inputs/flight.toml",
        aero_path,
    ]
    result = {
        path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path): _digest(path)
        for path in paths
    }
    # References may point into another input document. Include the resolved
    # configuration so those changes cannot bypass the ascent cache check.
    result["resolved_configuration"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    return result


def current_flight(recompute=False):
    """Return ``(history, summary, simulation)`` for the saved flight inputs.

    The private cache is disposable. Any changed input, missing output, bad hash
    or unreadable file triggers a fresh RK45 integration. Set ``recompute=True``
    to rerun even when the cache is valid.
    """
    folder = ROOT / ".cache/flight"
    record_path = folder / "fingerprint.json"
    expected = fingerprint()
    context = None
    if not recompute and record_path.exists():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            outputs = record.get("outputs", {})
            valid = (
                record.get("inputs") == expected
                and set(outputs) == {"trajectory.csv", "summary.json"}
                and all(
                    (folder / name).is_file() and _digest(folder / name) == digest
                    for name, digest in outputs.items()
                )
            )
            if valid:
                history = pd.read_csv(folder / "trajectory.csv")
                summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                simulation = Simulation(load_config())
                simulation.events = summary["events"]
                simulation.rows = history.to_dict("records")
                _, simulation.load_cases = summarize(history, simulation.events, simulation.c)
                context = history, summary, simulation
        except (OSError, ValueError, KeyError):
            # Interrupted saves and edited cache files are not usable results.
            context = None

    if context is None:
        context = run_trajectory()
        if fingerprint() != expected:
            raise RuntimeError("Flight inputs changed during integration. Run again before saving.")
        history, summary, _ = context
        folder.mkdir(parents=True, exist_ok=True)
        history.to_csv(folder / "trajectory.csv", index=False)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        record = {
            "inputs": expected,
            "outputs": {name: _digest(folder / name) for name in ("trajectory.csv", "summary.json")},
        }
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    context[2]._flight_fingerprint = expected
    return context
