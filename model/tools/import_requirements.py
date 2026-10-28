"""Read a synced verification workbook without changing it.

Usage: python model/tools/import_requirements.py "path/to/Verification Plan.xlsx"
Only requirements referenced by inputs/verification.toml are written. The full
workbook is read and validated first, and all source activities are retained
inside each selected requirement for honest whole-requirement closure checks.
Run OneDrive and confirm 'Backed up and synced' before importing. This script
does not control OneDrive or claim that a file was synchronized.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.verification import validate_scope

SHEETS = [
    "Tier 0",
    "1 LV Functional",
    "2 LV Performance",
    "3 Structural",
    "3 Propulsion",
    "3 Manufacturing",
    "3 Avionics",
    "3 Operations",
]


def import_workbook(path, destination=None, scope_file=None):
    """Write selected requirements only after validating exact scoped activities."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    requirements = []
    for sheet_name in SHEETS:
        sheet = workbook[sheet_name]
        columns = {
            str(cell.value).strip(): cell.column for cell in sheet[4] if cell.value
        }
        for row in sheet.iter_rows(min_row=5):

            def value(label):
                column = columns.get(label)
                return row[column - 1].value if column else None

            identifier = value("ID")
            if not isinstance(identifier, str) or not re.match(
                r"^[A-Z]+\.\d\.", identifier
            ):
                continue
            statement = str(value("Requirement") or "")
            activity_text = str(value("Verification Activity") or "")
            method_text = str(value("Verification Method") or "")
            methods = [
                method.strip()
                for method in re.split(r"[\n/]", method_text)
                if method.strip()
            ]
            activities = [
                activity.strip()
                for activity in activity_text.splitlines()
                if activity.strip()
            ]
            activity_cell = row[columns["Verification Activity"] - 1].coordinate
            statement_cell = row[columns["Requirement"] - 1].coordinate
            requirements.append(
                {
                    "id": identifier,
                    "domain": sheet_name,
                    "statement": statement,
                    "methods": methods,
                    "activities": activities,
                    "environment": value("Environment"),
                    "considerations": value("Special Considerations"),
                    "expected_artifacts": value("Verification Artifact"),
                    "parent_requirements": value("Parent Requirements"),
                    "notes": value("Notes"),
                    "open_parameters": sorted(
                        set(re.findall(r"TB[DR]-[A-Z0-9-]+", statement))
                    ),
                    "source_ref": f"{sheet_name}!{activity_cell}",
                    "statement_ref": f"{sheet_name}!{statement_cell}",
                    "source_file": path.name,
                    "source_sha256": digest,
                }
            )
    workbook.close()
    identifiers = [row["id"] for row in requirements]
    if not requirements or len(identifiers) != len(set(identifiers)):
        raise ValueError("Requirement IDs missing or duplicated; import rejected.")
    selected = validate_scope(requirements, scope_file)
    selected_ids = {row["requirement_id"] for row in selected}
    reference_count = len(requirements)
    requirements = [row for row in requirements if row["id"] in selected_ids]
    destination = Path(destination) if destination is not None else ROOT / "inputs/requirements/requirements.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(requirements, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "count": len(requirements),
        "assigned_activities": len(selected),
        "workbook_requirements_read": reference_count,
        "source_file": path.name,
        "sha256": digest,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--scope", type=Path, help="Explicit assignment TOML (default: inputs/verification.toml).")
    parser.add_argument("--destination", type=Path, help="Selected requirement JSON destination.")
    args = parser.parse_args()
    print(json.dumps(import_workbook(args.workbook, args.destination, args.scope), indent=2))
