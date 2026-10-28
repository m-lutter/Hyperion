"""Connect assigned verification activities to explicit calculations and evidence.

Calculations are design evidence, never automatic hardware qualification.  A
result can only be SCREEN_PASS when an explicit, sourced acceptance rule exists.
Inspection, demonstration and physical test records are checked separately.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from .inputs import load_checks, load_config

ROOT = Path(__file__).resolve().parents[1]
CALCULATIONS = (
    "flight_summary", "integrated_tmi", "lambert_window", "sourced_sizing", "thrust_sum",
)


def load_requirements():
    """Read the imported requirements, including their source references."""
    path = ROOT / "inputs/requirements/requirements.json"
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_id():
    """Content identity for the model, sourced requirements and editable inputs."""
    # Evidence records carry this identity, so including evidence would create a
    # circular hash. Generated outputs, examples and documentation are excluded.
    paths = sorted((ROOT / "model").glob("*.py"))
    paths += sorted((ROOT / "inputs").glob("*.toml"))
    aero = Path(load_config()["aero"]["file"])
    if not aero.is_absolute():
        aero = ROOT / aero
    aero = aero.resolve()
    paths += [
        ROOT / "inputs/requirements/requirements.json",
        ROOT / "inputs/requirements/mission_cases.json",
        aero,
    ]
    manifest = {
        path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def catalog():
    """Return the five assigned calculations, without scanning module registries.

    The empty second item preserves the historical tuple interface. Activity
    routing comes exclusively from inputs/verification.toml.
    """
    from .flight_evidence import flight_summary, sourced_sizing
    from .mission import integrated_tmi, lambert_window
    from .propulsion import thrust_sum

    return {
        "flight_summary": flight_summary,
        "integrated_tmi": integrated_tmi,
        "lambert_window": lambert_window,
        "sourced_sizing": sourced_sizing,
        "thrust_sum": thrust_sum,
    }, {}


def load_scope(scope_file=None):
    """Read the explicit assignment list; source prose never selects a route."""
    path = Path(scope_file) if scope_file is not None else ROOT / "inputs/verification.toml"
    with path.open("rb") as stream:
        scope = tomllib.load(stream)
    if not isinstance(scope.get("description"), str) or not scope["description"].strip():
        raise ValueError("Verification scope needs a nonempty description.")
    entries = scope.get("activities")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Verification scope needs at least one [[activities]] entry.")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each verification scope entry must be a table.")
        for key in ("activity_id", "calculation", "expected_activity", "artifact", "coverage"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"Verification scope entry needs nonempty {key}.")
        identifier = entry["activity_id"]
        if identifier in seen:
            raise ValueError(f"Duplicate scoped activity ID: {identifier}")
        seen.add(identifier)
        if entry["calculation"] not in CALCULATIONS:
            raise ValueError(
                f"Unknown calculation route {entry['calculation']!r} for {identifier}."
            )
    return scope


def validate_scope(requirements, scope_file=None):
    """Resolve exact activity IDs and reject source wording that has changed.

    This also validates a full workbook before the importer writes its selected
    requirements. Unassigned workbook activities are retained as closure context
    inside those requirement records, but are never added to the run.
    """
    scope = load_scope(scope_file)
    source = {}
    seen_requirements = set()
    for requirement in requirements:
        identifier = requirement["id"]
        if identifier in seen_requirements:
            raise ValueError(f"Duplicate source requirement ID: {identifier}")
        seen_requirements.add(identifier)
        for number, activity in enumerate(requirement["activities"], 1):
            source[f"{identifier}.A{number:02d}"] = requirement, activity
    rows = []
    for entry in scope["activities"]:
        identifier = entry["activity_id"]
        if identifier not in source:
            raise ValueError(f"Unknown scoped activity ID: {identifier}")
        requirement, activity = source[identifier]
        if entry["expected_activity"] != activity:
            raise ValueError(
                f"Scoped activity wording mismatch for {identifier}: "
                f"expected {entry['expected_activity']!r}; source is {activity!r}. "
                "Review the assignment against the current workbook."
            )
        rows.append({
            "activity_id": identifier,
            "requirement_id": requirement["id"],
            "domain": requirement["domain"],
            "activity": activity,
            "kernel": entry["calculation"],
            "artifact": entry["artifact"],
            "coverage": entry["coverage"],
            "methods": " / ".join(requirement["methods"]),
            "source": requirement["source_ref"],
            "statement": requirement["statement"],
            "open_parameters": ", ".join(requirement["open_parameters"]),
            "source_file": requirement.get("source_file", ""),
            "source_sha256": requirement.get("source_sha256", ""),
        })
    return rows


def activities(scope_file=None):
    """Return only assigned rows, with exact current source wording and coverage."""
    return pd.DataFrame(validate_scope(load_requirements(), scope_file))


def input_contracts(scope_file=None):
    """List required arguments for the calculations selected in the scope."""
    functions, _ = catalog()
    rows = []
    for key in dict.fromkeys(activities(scope_file).kernel):
        function = functions[key]
        required_inputs = []
        for name, parameter in inspect.signature(function).parameters.items():
            has_default = parameter.default is not inspect.Parameter.empty
            accepts_many = parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            if not has_default and not accepts_many:
                required_inputs.append(name)
        rows.append(
            {
                "calculation": key,
                "required_inputs": ", ".join(required_inputs),
                "description": (inspect.getdoc(function) or "").split("\n")[0],
            }
        )
    return pd.DataFrame(rows)


def activity_index(scope_file=None):
    """Link each assigned activity to its explicit calculation route."""
    table = activities(scope_file)
    functions, _ = catalog()
    code_routes = []
    route_types = []
    for kernel in table.kernel:
        if isinstance(kernel, str):
            function = functions[kernel]
            code_routes.append(f"{function.__module__}.{function.__name__}")
            route_types.append("calculation")
        else:
            code_routes.append("model.verification.check_evidence / measured_limits")
            route_types.append("external evidence validation")
    table["code_route"] = code_routes
    table["route_type"] = route_types
    return table


def _plain(value):
    """Convert NumPy results to JSON values; nonfinite numbers become null."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def check_evidence(record, root=ROOT, expected_configuration=None):
    """Validate a reviewed external record and its file hash, not its engineering truth.

    Required fields: file, sha256, reviewer, date, result, configuration_id,
    acceptance_criteria, scope. The reviewer must explicitly cover this activity.
    Paths must stay within the repository; a missing record stays open.
    """
    required_fields = (
        "file",
        "sha256",
        "reviewer",
        "date",
        "result",
        "configuration_id",
        "acceptance_criteria",
        "scope",
    )
    missing = [key for key in required_fields if not record.get(key)]
    if missing:
        return "EVIDENCE_REQUIRED", "Missing record fields: " + ", ".join(missing)
    if expected_configuration is not None:
        if record["configuration_id"] != expected_configuration:
            return (
                "STALE_EVIDENCE",
                "Reviewed configuration differs from the current model/data/requirements baseline.",
            )
    root = Path(root).resolve()
    path = (root / record["file"]).resolve()
    if not path.is_relative_to(root):
        return "INVALID_EVIDENCE", "Evidence file must remain within the repository."
    if not path.is_file():
        return "EVIDENCE_REQUIRED", f"Missing evidence file: {record['file']}"
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != record["sha256"]:
        return "INVALID_EVIDENCE", "File hash differs from the reviewed evidence."
    if record["result"] not in ("pass", "fail"):
        return "INVALID_EVIDENCE", "Reviewed result must be pass or fail."
    status = "REVIEWED_PASS" if record["result"] == "pass" else "REVIEWED_FAIL"
    return status, record["scope"]


def measured_limits(csv_path):
    """Evaluate a supplied measurement/result CSV against explicit row limits.

    Columns: case_id, quantity, value, unit, lower_limit, upper_limit, criterion_source.
    At least one finite bound is required per row. Missing/non-numeric values and
    duplicate case/quantity rows are invalid, never successful zero measurements.
    The responsible reviewer establishes required test-case coverage separately.
    """
    frame = pd.read_csv(csv_path)
    required = [
        "case_id",
        "quantity",
        "value",
        "unit",
        "lower_limit",
        "upper_limit",
        "criterion_source",
    ]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError(
            "A nonempty measurement table with all required columns is needed."
        )
    if frame.duplicated(["case_id", "quantity"]).any():
        raise ValueError(
            "Duplicate case/quantity measurements require explicit distinct case IDs."
        )
    for key in ["case_id", "quantity", "unit", "criterion_source"]:
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing measurement metadata: {key}")
    for key in ["value", "lower_limit", "upper_limit"]:
        frame[key] = pd.to_numeric(frame[key], errors="raise")
    missing_limits = frame[["lower_limit", "upper_limit"]].isna().all(axis=1)
    if not np.isfinite(frame.value).all() or missing_limits.any():
        raise ValueError(
            "Every measurement requires a finite value and at least one limit."
        )
    for key in ["lower_limit", "upper_limit"]:
        if np.isinf(frame[key].dropna()).any():
            raise ValueError("Bounds must be finite when supplied.")
    if (frame.lower_limit > frame.upper_limit).any():
        raise ValueError("Lower bound exceeds upper bound.")
    above_lower_limit = frame.lower_limit.isna() | (frame.value >= frame.lower_limit)
    below_upper_limit = frame.upper_limit.isna() | (frame.value <= frame.upper_limit)
    frame["within_limits"] = above_lower_limit & below_upper_limit
    return frame


def acceptance(metrics, rules):
    """Apply explicit scalar rules with units and a source; no inferred thresholds."""
    outcomes = []
    for rule in rules:
        if not rule.get("source") or not rule.get("units"):
            raise ValueError("Every acceptance rule needs units and a source.")
        value = metrics
        for key in rule["metric"].split("."):
            value = value[key]
        if value is None or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"No finite scalar for {rule['metric']}")
        operators = {
            "<=": lambda value, limit: value <= limit,
            ">=": lambda value, limit: value >= limit,
            "<": lambda value, limit: value < limit,
            ">": lambda value, limit: value > limit,
            "==": lambda value, limit: value == limit,
        }
        outcomes.append(bool(operators[rule["operator"]](value, rule["limit"])))
    return all(outcomes) if outcomes else None


def evaluate(input_file=None, evidence_file=None, context=None, scope_file=None):
    """Run only assigned activities, retaining acceptance and evidence checks.

    Inputs are keyed by activity ID (specific) or calculation name (shared).
    A shared result is cached only for the identical kernel/arguments/criteria.
    Context carries the one integrated flight run; never read stale result CSVs.
    """
    selected = activities(scope_file)
    functions, _ = catalog()
    inputs = load_checks(input_file)
    evidence_path = Path(evidence_file or ROOT / "inputs/evidence/records.json")
    evidence = {}
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    rows = []
    details = {}
    cache = {}
    current_baseline = baseline_id()
    for activity in selected.to_dict("records"):
        activity_id = activity["activity_id"]
        kernel = activity["kernel"]
        record = evidence.get(activity_id)
        if record:
            status, note = check_evidence(
                record, expected_configuration=current_baseline
            )
            # A record cannot close a different activity or unknown configuration.
            if record.get("activity_id") != activity_id:
                status, note = (
                    "INVALID_EVIDENCE",
                    "Activity ID does not match the reviewed record.",
                )
            if record.get("activity_text") != activity["activity"]:
                status, note = (
                    "STALE_EVIDENCE",
                    "Reviewed activity wording differs from the current requirement activity.",
                )
            rows.append({**activity, "status": status, "detail": note})
            continue
        if not isinstance(kernel, str) or not kernel:
            rows.append(
                {
                    **activity,
                    "status": "EVIDENCE_REQUIRED",
                    "detail": "Add the CAD/FEA, specialist analysis, inspection, demonstration or test record for this activity; see inputs/evidence/README.md.",
                }
            )
            continue

        # Activity-specific inputs override inputs shared by a calculation.
        spec = inputs.get(activity_id, inputs.get(kernel, {}))
        kwargs = dict(spec.get("inputs", {}))
        function = functions[kernel]
        signature = inspect.signature(function)
        if "context" in signature.parameters:
            kwargs["context"] = context

        missing = []
        for name, parameter in signature.parameters.items():
            has_default = parameter.default is not inspect.Parameter.empty
            accepts_many = parameter.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
            if not has_default and not accepts_many and kwargs.get(name) is None:
                missing.append(name)
        if missing:
            rows.append(
                {**activity, "status": "MISSING_INPUTS", "detail": ", ".join(missing)}
            )
            continue

        cache_key = json.dumps([kernel, spec], sort_keys=True)
        try:
            if cache_key not in cache:
                result = _plain(function(**kwargs))
                passed = acceptance(result, spec.get("acceptance", []))
                cache[cache_key] = result, passed
            result, passed = cache[cache_key]
            details[activity_id] = result
            if passed is None:
                status = "CALCULATED"
            elif passed:
                status = "SCREEN_PASS"
            else:
                status = "SCREEN_FAIL"
            note = spec.get(
                "basis",
                "Calculation only; activity-specific acceptance and model validation remain required.",
            )
            if spec.get("example_only", False):
                status, note = (
                    "EXAMPLE_ONLY",
                    "Illustrative inputs; this is not Hyperion verification evidence.",
                )
        except (
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as error:
            status, note = "INVALID_INPUTS", str(error)
        rows.append({**activity, "status": status, "detail": note})
    return pd.DataFrame(rows), details


def requirement_status(results, scope_file=None):
    """Summarize assigned coverage separately from whole-requirement closure.

    Only requirements represented by assigned results appear. Even reviewed
    assigned work cannot close a requirement with other source activities.
    """
    selected = activities(scope_file)
    results = results[results.activity_id.isin(selected.activity_id)]
    requirements = {row["id"]: row for row in load_requirements()}
    rows = []
    for identifier in results.requirement_id.drop_duplicates():
        if identifier not in requirements:
            raise ValueError(f"Unknown result requirement ID: {identifier}")
        requirement = requirements[identifier]
        group = results[results.requirement_id == identifier]
        assigned = set(selected.loc[selected.requirement_id == identifier, "activity_id"])
        complete_assigned = (
            set(group.activity_id) == assigned
            and len(group) == len(assigned)
            and bool(assigned)
        )
        failed = group.status.isin(
            [
                "SCREEN_FAIL",
                "REVIEWED_FAIL",
                "INVALID_EVIDENCE",
                "INVALID_INPUTS",
            ]
        ).any()
        expected = {
            f"{requirement['id']}.A{number:02d}"
            for number in range(1, len(requirement["activities"]) + 1)
        }
        # Matching IDs alone would overlook duplicate records.
        complete = (
            set(group.activity_id) == expected
            and len(group) == len(expected)
            and bool(expected)
        )
        reviewed = complete and group.status.eq("REVIEWED_PASS").all()
        status = "FAILED_OR_INVALID" if failed else "OPEN"
        if reviewed and not requirement["open_parameters"]:
            status = "REVIEWED_COMPLETE"
        calculated = group.status.isin(["CALCULATED", "SCREEN_PASS", "SCREEN_FAIL"])
        rows.append(
            {
                "id": requirement["id"],
                "domain": requirement["domain"],
                "status": status,
                "activities": len(group),
                "assigned_activities": len(assigned),
                "complete_assigned_set": complete_assigned,
                "assigned_reviewed_complete": bool(
                    complete_assigned and group.status.eq("REVIEWED_PASS").all()
                ),
                "expected_activities": len(expected),
                "complete_activity_set": complete,
                "calculated": int(calculated.sum()),
                "open_parameters": ", ".join(requirement["open_parameters"]),
                "source": requirement["source_ref"],
            }
        )
    return pd.DataFrame(rows)


def export(results, details, output_dir, scope_file=None):
    """Save only assigned results, with long calculation histories in CSV files."""
    selected_ids = set(activities(scope_file).activity_id)
    results = results[results.activity_id.isin(selected_ids)]
    result_ids = set(results.activity_id)
    details = {key: value for key, value in details.items() if key in result_ids}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "activity_results.csv", index=False)
    requirement_status(results, scope_file).to_csv(output / "requirement_status.csv", index=False)
    compact = {}
    histories = {}
    for activity_id, result in details.items():
        result = dict(result)
        if isinstance(result.get("history"), list):
            history = result.pop("history")
            digest = hashlib.sha256(
                json.dumps(history, sort_keys=True).encode()
            ).hexdigest()[:12]
            relative = f"histories/{digest}.csv"
            if digest not in histories:
                (output / "histories").mkdir(exist_ok=True)
                pd.DataFrame(history).to_csv(output / relative, index=False)
                histories[digest] = relative
            result["history_csv"] = relative
        compact[activity_id] = result
    (output / "calculation_results.json").write_text(
        json.dumps(_plain(compact), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output
