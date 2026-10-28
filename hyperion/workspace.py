"""Reuse a flight run only when model, data, configuration and outputs match."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .trajectory import Simulation, load_config, run_trajectory, summarize
from .verification import ROOT


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    """Hash the model, selected configuration and aerodynamic lookup table."""
    return {
        str(p.relative_to(ROOT)): _digest(p)
        for p in [
            ROOT / "hyperion/trajectory.py",
            ROOT / "config/trajectory.json",
            ROOT / "data/aero_lookup.csv",
        ]
    }


def current_flight(recompute=False):
    """Return ``(history, summary, simulation)`` for the current ascent inputs.

    Reuse saved results only if every input and output file still matches its
    recorded hash. Set ``recompute=True`` to force a fresh integration.
    """
    folder = ROOT / "results/flight_cache"
    record_path = folder / "fingerprint.json"
    expected = fingerprint()
    if not recompute and record_path.exists():
        record = json.loads(record_path.read_text())
        inputs_match = record.get("inputs") == expected
        outputs_match = inputs_match and all(
            (folder / name).is_file() and _digest(folder / name) == digest
            for name, digest in record.get("outputs", {}).items()
        )
        outputs_complete = outputs_match and set(record.get("outputs", {})) == {
            "trajectory.csv",
            "summary.json",
        }
        if inputs_match and outputs_match and outputs_complete:
            history = pd.read_csv(folder / "trajectory.csv")
            summary = json.loads((folder / "summary.json").read_text())
            simulation = Simulation(load_config())
            simulation.events = summary["events"]
            simulation.rows = history.to_dict("records")
            _, simulation.load_cases = summarize(
                history, simulation.events, simulation.c
            )
            return history, summary, simulation

    # A missing or changed file invalidates the cache as a whole.
    context = run_trajectory()
    history, summary, _ = context
    folder.mkdir(parents=True, exist_ok=True)
    history.to_csv(folder / "trajectory.csv", index=False)
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    record = {
        "inputs": expected,
        "outputs": {
            name: _digest(folder / name) for name in ["trajectory.csv", "summary.json"]
        },
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return context
