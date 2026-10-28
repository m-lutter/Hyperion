"""Validate the assigned activity-to-code index without changing saved runs.

Use --write-cache to inspect CSV copies under .cache/verification_index.
The Verification notebook owns the published indexes in outputs/current.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.verification import activity_index, input_contracts


def main(write_cache=False, scope_file=None):
    table = activity_index(scope_file)
    contracts = input_contracts(scope_file)
    if write_cache:
        output = ROOT / ".cache/verification_index"
        output.mkdir(parents=True, exist_ok=True)
        table.to_csv(output / "activities.csv", index=False)
        contracts.to_csv(output / "calculation_inputs.csv", index=False)
    assigned = table.kernel.notna().sum()
    print(f"Indexed {len(table)} assigned activities using {len(contracts)} calculations; {assigned} routes validated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate without writing files (default).")
    mode.add_argument("--write-cache", action="store_true", help="Also write disposable index CSVs under .cache/.")
    parser.add_argument("--scope", type=Path, help="Explicit assignment TOML (default: inputs/verification.toml).")
    args = parser.parse_args()
    main(write_cache=args.write_cache, scope_file=args.scope)
