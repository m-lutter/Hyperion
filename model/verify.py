"""Run the shared ascent/departure and every registered verification activity.

    python -m model.verify
    python -m model.verify --recompute --out outputs/current
"""

import argparse
import json
from pathlib import Path

from .inputs import load_checks
from .mission import integrated_tmi
from .outputs import DEFAULT_OUTPUT, input_identity, save_flight, save_verification
from .verification import evaluate
from .workspace import current_flight


def run(output_dir=DEFAULT_OUTPUT, recompute=False):
    """Use one ascent, evaluate all activities, and save traceable readable results."""
    identity = input_identity()
    context = current_flight(recompute=recompute)
    # Run the selected departure even when verification uses reviewed evidence
    # or activity-specific overrides. This matches the Simulation notebook.
    departure = None
    if any(event["event"] == "orbital_speed_cutoff" for event in context[1]["events"]):
        settings = load_checks()["integrated_tmi"]["inputs"]
        departure = integrated_tmi(context=context, **settings)
    results, details = evaluate(context=context)
    save_flight(context, departure=departure, output_dir=output_dir, expected_inputs=identity)
    output = save_verification(results, details, output_dir=output_dir, expected_inputs=identity)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    return context, results, details, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recompute", action="store_true", help="Ignore the matching private flight cache.")
    args = parser.parse_args()
    _, results, _, _ = run(args.out, args.recompute)
    print(f"Saved {len(results)} activity results to {args.out.resolve()}")
    print(f"Read {(args.out / 'summary.md').resolve()}")
