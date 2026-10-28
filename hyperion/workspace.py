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
    return {str(p.relative_to(ROOT)): _digest(p) for p in
            [ROOT / "hyperion/trajectory.py", ROOT / "config/trajectory.json", ROOT / "data/aero_lookup.csv"]}


def current_flight(recompute=False):
    """Return (history, summary, Simulation), recomputing if any fingerprint differs."""
    folder = ROOT / "results/flight_cache"
    record_path = folder / "fingerprint.json"
    expected = fingerprint()
    if not recompute and record_path.exists():
        record = json.loads(record_path.read_text())
        valid = record.get("inputs") == expected and all(
            (folder / name).is_file() and _digest(folder / name) == digest
            for name, digest in record.get("outputs", {}).items())
        if valid and set(record.get("outputs", {})) == {"trajectory.csv", "summary.json"}:
            frame = pd.read_csv(folder / "trajectory.csv")
            summary = json.loads((folder / "summary.json").read_text())
            sim = Simulation(load_config())
            sim.events = summary["events"]
            sim.rows = frame.to_dict("records")
            _, sim.load_cases = summarize(frame, sim.events, sim.c)
            return frame, summary, sim
    context = run_trajectory()
    folder.mkdir(parents=True, exist_ok=True)
    context[0].to_csv(folder / "trajectory.csv", index=False)
    (folder / "summary.json").write_text(json.dumps(context[1], indent=2), encoding="utf-8")
    record = {"inputs": expected, "outputs": {name: _digest(folder / name)
              for name in ["trajectory.csv", "summary.json"]}}
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return context
