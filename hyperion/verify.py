"""Run the integrated model and every registered verification activity.

python -m hyperion.verify --out results/current
"""

import argparse
import hashlib
import json
from pathlib import Path

from .trajectory import load_config, run_trajectory
from .verification import ROOT, evaluate, export, requirement_status


def run(output_dir=ROOT / "results/current"):
    """Run the shared trajectory, evaluate activities, and save the evidence tables."""
    config = load_config()
    config["solver"]["verbose"] = False
    context = run_trajectory(config)
    table, details = evaluate(context=context)
    output = export(table, details, output_dir)

    trajectory, summary, _ = context
    trajectory.to_csv(output / "trajectory.csv", index=False)
    (output / "trajectory_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    requirements = requirement_status(table)
    manifest = {
        "requirements": len(requirements),
        "activities": len(table),
        "activity_status": {
            str(status): int(count)
            for status, count in table.status.value_counts().items()
        },
        "requirement_status": {
            str(status): int(count)
            for status, count in requirements.status.value_counts().items()
        },
        "requirements_sha256": hashlib.sha256(
            (ROOT / "data/requirements.json").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            (ROOT / "config/trajectory.json").read_bytes()
        ).hexdigest(),
        "aero_sha256": hashlib.sha256(
            (ROOT / "data/aero_lookup.csv").read_bytes()
        ).hexdigest(),
        "scope": "Design screens and evidence completeness; not flight qualification.",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return context, table, details, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/current")
    args = parser.parse_args()
    _, _, _, manifest = run(args.out)
    print(json.dumps(manifest, indent=2))
