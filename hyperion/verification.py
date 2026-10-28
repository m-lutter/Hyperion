"""Requirement/activity traceability and conservative evidence evaluation.

Calculations are design evidence, never automatic hardware qualification.  A
result can only be SCREEN_PASS when an explicit, sourced acceptance rule exists.
Inspection, demonstration and physical test records are checked separately.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_requirements():
    return json.loads((ROOT / "data/requirements.json").read_text(encoding="utf-8"))


def baseline_id():
    """Content identity for the model, sourced requirements and editable inputs."""
    paths = sorted((ROOT / 'hyperion').glob('*.py')) + sorted((ROOT / 'config').glob('*.json'))
    paths += [ROOT / 'data/requirements.json', ROOT / 'data/aero_lookup.csv']
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths}
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()


def catalog():
    """Explicit activity-to-calculation assignments; no text similarity scoring."""
    functions, assignments = {}, {}
    for name in ("mission", "propulsion", "structures", "systems", "flight_evidence"):
        module = importlib.import_module(f"hyperion.{name}")
        for key, function in module.FUNCTIONS.items():
            if key in functions:
                raise ValueError(f"Duplicate calculation key: {key}")
            functions[key] = function
        for activity, key in module.KERNEL_ACTIVITIES.items():
            # A later flight-specific assignment takes precedence over a general model.
            assignments[activity.rstrip(".").strip().casefold()] = key
    return functions, assignments


def activities():
    """One row per activity, retaining each requirement's original wording."""
    _, assignments = catalog()
    rows = []
    for requirement in load_requirements():
        for i, activity in enumerate(requirement["activities"], 1):
            kernel = assignments.get(activity.rstrip(".").strip().casefold())
            rows.append(dict(activity_id=f"{requirement['id']}.A{i:02d}",
                             requirement_id=requirement["id"], domain=requirement["domain"],
                             activity=activity, kernel=kernel, methods=" / ".join(requirement["methods"]),
                             source=requirement["source_ref"], statement=requirement["statement"],
                             open_parameters=", ".join(requirement["open_parameters"])))
    return pd.DataFrame(rows)


def input_contracts():
    functions, _ = catalog()
    return pd.DataFrame([
        {"calculation": key, "required_inputs": ", ".join(
            name for name, p in inspect.signature(fn).parameters.items()
            if p.default is inspect.Parameter.empty and p.kind not in
            (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)),
         "description": (inspect.getdoc(fn) or "").split("\n")[0]}
        for key, fn in functions.items()
    ])


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
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
    needed = ("file", "sha256", "reviewer", "date", "result", "configuration_id",
              "acceptance_criteria", "scope")
    missing = [key for key in needed if not record.get(key)]
    if missing:
        return "EVIDENCE_REQUIRED", "Missing record fields: " + ", ".join(missing)
    if expected_configuration is not None and record['configuration_id'] != expected_configuration:
        return "STALE_EVIDENCE", "Reviewed configuration differs from the current model/data/requirements baseline."
    root = Path(root).resolve()
    path = (root / record["file"]).resolve()
    if not path.is_relative_to(root):
        return "INVALID_EVIDENCE", "Evidence file must remain within the repository."
    if not path.is_file():
        return "EVIDENCE_REQUIRED", f"Missing evidence file: {record['file']}"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        return "INVALID_EVIDENCE", "File hash differs from the reviewed evidence."
    if record["result"] not in ("pass", "fail"):
        return "INVALID_EVIDENCE", "Reviewed result must be pass or fail."
    return ("REVIEWED_PASS" if record["result"] == "pass" else "REVIEWED_FAIL"), record["scope"]


def measured_limits(csv_path):
    """Evaluate a supplied measurement/result CSV against explicit row limits.

    Columns: case_id, quantity, value, unit, lower_limit, upper_limit, criterion_source.
    At least one finite bound is required per row. Missing/non-numeric values and
    duplicate case/quantity rows are invalid, never successful zero measurements.
    The responsible reviewer establishes required test-case coverage separately.
    """
    frame = pd.read_csv(csv_path)
    required = ['case_id','quantity','value','unit','lower_limit','upper_limit','criterion_source']
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError('A nonempty measurement table with all required columns is needed.')
    if frame.duplicated(['case_id','quantity']).any():
        raise ValueError('Duplicate case/quantity measurements require explicit distinct case IDs.')
    for key in ['case_id','quantity','unit','criterion_source']:
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing measurement metadata: {key}')
    for key in ['value','lower_limit','upper_limit']:
        frame[key] = pd.to_numeric(frame[key],errors='raise')
    if not np.isfinite(frame.value).all() or frame[['lower_limit','upper_limit']].isna().all(axis=1).any():
        raise ValueError('Every measurement requires a finite value and at least one limit.')
    for key in ['lower_limit','upper_limit']:
        if np.isinf(frame[key].dropna()).any():
            raise ValueError('Bounds must be finite when supplied.')
    if (frame.lower_limit > frame.upper_limit).any():
        raise ValueError('Lower bound exceeds upper bound.')
    frame['within_limits'] = ((frame.lower_limit.isna() | (frame.value>=frame.lower_limit)) &
                              (frame.upper_limit.isna() | (frame.value<=frame.upper_limit)))
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
        operators = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
                     "<": lambda a, b: a < b, ">": lambda a, b: a > b,
                     "==": lambda a, b: a == b}
        outcomes.append(bool(operators[rule["operator"]](value, rule["limit"])))
    return all(outcomes) if outcomes else None


def evaluate(input_file=None, evidence_file=None, context=None):
    """Run every activity, reporting missing inputs and external evidence honestly.

    Inputs are keyed by activity ID (specific) or calculation name (shared).
    A shared result is cached only for the identical kernel/arguments/criteria.
    Context carries the one integrated flight run; never read stale result CSVs.
    """
    functions, _ = catalog()
    path = Path(input_file or ROOT / "config/verification_inputs.json")
    inputs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ep = Path(evidence_file or ROOT / "evidence/records.json")
    evidence = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else {}
    rows, details, cache = [], {}, {}
    current_baseline = baseline_id()
    for activity in activities().to_dict("records"):
        aid, kernel = activity["activity_id"], activity["kernel"]
        record = evidence.get(aid)
        if record:
            status, note = check_evidence(record, expected_configuration=current_baseline)
            # A record cannot close a different activity or unknown configuration.
            if record.get("activity_id") != aid:
                status, note = "INVALID_EVIDENCE", "Activity ID does not match the reviewed record."
            if record.get('activity_text') != activity['activity']:
                status, note = "STALE_EVIDENCE", "Reviewed activity wording differs from the current requirement activity."
            rows.append({**activity, "status": status, "detail": note})
            continue
        if not isinstance(kernel, str) or not kernel:
            rows.append({**activity, "status": "EVIDENCE_REQUIRED", "detail":
                "Import the activity-specific CAD/FEA, specialist-model, inspection, demonstration or test record; see evidence/README.md."})
            continue
        spec = inputs.get(aid, inputs.get(kernel, {}))
        kwargs = dict(spec.get("inputs", {}))
        fn = functions[kernel]
        signature = inspect.signature(fn)
        if "context" in signature.parameters:
            kwargs["context"] = context
        missing = [n for n, p in signature.parameters.items()
                   if p.default is inspect.Parameter.empty and
                   p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
                   and (n not in kwargs or kwargs[n] is None)]
        if missing:
            rows.append({**activity, "status": "MISSING_INPUTS", "detail": ", ".join(missing)})
            continue
        cache_key = json.dumps([kernel, spec], sort_keys=True)
        try:
            if cache_key not in cache:
                result = _plain(fn(**kwargs))
                passed = acceptance(result, spec.get("acceptance", []))
                cache[cache_key] = result, passed
            result, passed = cache[cache_key]
            details[aid] = result
            status = "CALCULATED" if passed is None else ("SCREEN_PASS" if passed else "SCREEN_FAIL")
            note = spec.get("basis", "Calculation only; activity-specific acceptance and model validation remain required.")
            if spec.get("example_only", False):
                status, note = "EXAMPLE_ONLY", "Illustrative inputs; this is not Hyperion verification evidence."
        except (ValueError, KeyError, TypeError, RuntimeError, np.linalg.LinAlgError) as error:
            status, note = "INVALID_INPUTS", str(error)
        rows.append({**activity, "status": status, "detail": note})
    return pd.DataFrame(rows), details


def requirement_status(results):
    """Never promote a calculation screen to closed requirement verification."""
    rows = []
    for requirement in load_requirements():
        group = results[results.requirement_id == requirement["id"]]
        failed = group.status.isin(["SCREEN_FAIL", "REVIEWED_FAIL", "INVALID_EVIDENCE", "INVALID_INPUTS"]).any()
        expected = {f"{requirement['id']}.A{i:02d}" for i in range(1,len(requirement['activities'])+1)}
        complete = set(group.activity_id) == expected and len(group) == len(expected) and bool(expected)
        reviewed = complete and group.status.eq("REVIEWED_PASS").all()
        status = "FAILED_OR_INVALID" if failed else "OPEN"
        if reviewed and not requirement["open_parameters"]:
            status = "REVIEWED_COMPLETE"
        rows.append({"id": requirement["id"], "domain": requirement["domain"],
                     "status": status, "activities": len(group),
                     "expected_activities": len(expected), "complete_activity_set": complete,
                     "calculated": int(group.status.isin(["CALCULATED", "SCREEN_PASS", "SCREEN_FAIL"]).sum()),
                     "open_parameters": ", ".join(requirement["open_parameters"]),
                     "source": requirement["source_ref"]})
    return pd.DataFrame(rows)


def export(results, details, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "activity_results.csv", index=False)
    requirement_status(results).to_csv(output / "requirement_status.csv", index=False)
    compact, histories = {}, {}
    for aid, result in details.items():
        result = dict(result)
        if isinstance(result.get("history"), list):
            history = result.pop("history")
            digest = hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()[:12]
            relative = f"histories/{digest}.csv"
            if digest not in histories:
                (output / "histories").mkdir(exist_ok=True)
                pd.DataFrame(history).to_csv(output / relative, index=False)
                histories[digest] = relative
            result["history_csv"] = relative
        compact[aid] = result
    (output / "calculation_results.json").write_text(json.dumps(_plain(compact), indent=2, allow_nan=False), encoding="utf-8")
    return output
