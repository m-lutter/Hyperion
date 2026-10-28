"""Save readable results with the exact inputs that produced them.

All notebooks share outputs/current. A changed baseline starts a clean current
folder and archives the previous files in .cache/output_history. Each summary
states which notebooks have been run for the current inputs.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import warnings

import pandas as pd

from .inputs import choices_table, input_tables, load_checks, load_config, load_study
from .verification import (
    ROOT, _plain, acceptance, activity_index, baseline_id, export, input_contracts,
)

DEFAULT_OUTPUT = ROOT / "outputs/current"
SECTION_NAMES = {
    "flight": "Simulation",
    "engineering": "Engineering checks",
    "verification": "Verification",
}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_identity():
    """Capture this before calculation; pass it as expected_inputs when saving.

    The baseline excludes reviewed evidence to avoid a circular configuration ID.
    The full input identity includes evidence, so changed records invalidate old
    verification results without changing the configuration they were reviewing.
    """
    inputs = {
        path.relative_to(ROOT).as_posix(): _hash(path)
        for path in sorted((ROOT / "inputs").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    aero = Path(load_config()["aero"]["file"])
    if not aero.is_absolute():
        aero = ROOT / aero
    aero = aero.resolve()
    identity = {
        "baseline_id": baseline_id(), "files": inputs,
        "selected_aerodynamics": {
            "path": aero.relative_to(ROOT).as_posix() if aero.is_relative_to(ROOT) else str(aero),
            "sha256": _hash(aero),
        },
    }
    identity["input_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return identity


def _write_json(path, value):
    path.write_text(json.dumps(_plain(value), indent=2, allow_nan=False), encoding="utf-8")


def _name(value):
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()
    if not name:
        raise ValueError("An output needs a descriptive name.")
    return name


def _markdown(frame):
    """Small Markdown table without an optional tabulate dependency."""
    def text(value):
        if isinstance(value, float):
            value = f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(map(text, frame.columns)) + " |"]
    rows += ["| " + " | ".join("---" for _ in frame.columns) + " |"]
    rows += ["| " + " | ".join(map(text, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def _snapshot(output, identity):
    folder = output / "inputs_used"
    folder.mkdir(parents=True, exist_ok=True)
    for relative in identity["files"]:
        source = ROOT / relative
        destination = folder / Path(relative).relative_to("inputs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    selected_aero = identity.get("selected_aerodynamics")
    if selected_aero:
        aero = (ROOT / selected_aero["path"]).resolve()
        if not aero.is_relative_to(ROOT / "inputs"):
            shutil.copy2(aero, folder / "selected_aerodynamics.csv")
    # Resolved values expose references and derived settings in a readable table.
    groups = []
    for name, table in input_tables().items():
        table = table.copy()
        table.insert(0, "Group", name)
        groups.append(table)
    pd.concat(groups, ignore_index=True).to_csv(folder / "inputs.csv", index=False)
    choices_table().to_csv(folder / "choices.csv", index=False)
    return {
        path.relative_to(output).as_posix(): _hash(path)
        for path in sorted(folder.rglob("*")) if path.is_file()
    }


def _prepare(output_dir, expected_inputs):
    identity = input_identity()
    if expected_inputs is not None and expected_inputs != identity:
        raise RuntimeError("Inputs or model code changed after this calculation started. Rerun before saving.")
    output = Path(output_dir or DEFAULT_OUTPUT).resolve()
    if output == ROOT or output in ROOT.parents or any(
        output.is_relative_to(ROOT / name) for name in ("inputs", "model", "tests")
    ):
        raise ValueError("Choose an output directory, separate from source files.")
    manifest_path = output / "run_manifest.json"
    old = {}
    if manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    reason = None
    if output.exists() and any(output.iterdir()):
        if old.get("identity") != identity:
            reason = "The inputs or model changed; earlier sections must be rerun."
        elif not old.get("snapshot_files") or any(
            not (output / path).is_file() or _hash(output / path) != digest
            for path, digest in old["snapshot_files"].items()
        ):
            reason = "The saved input snapshot changed or was missing; earlier sections must be rerun."
        elif any(
            not (output / path).is_file() or _hash(output / path) != digest
            for section in old.get("sections", {}).values()
            for path, digest in section.get("files", {}).items()
        ):
            reason = "A saved result changed or was missing; earlier sections must be rerun."
    if reason:
        archive_root = (
            ROOT / ".cache/output_history"
            if output == DEFAULT_OUTPUT.resolve()
            else output.parent / ".output_history"
        ).resolve()
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive = (archive_root / f"{output.name}_{stamp}").resolve()
        if not archive.is_relative_to(archive_root) or archive.is_relative_to(output):
            raise ValueError("Invalid output archive location.")
        # Both absolute paths have been checked before moving this output folder.
        shutil.move(str(output), str(archive))
        warnings.warn(reason, stacklevel=3)
        old = {"notice": reason, "previous_results": str(archive)}
    output.mkdir(parents=True, exist_ok=True)
    for name in ("plots", "tables"):
        (output / name).mkdir(exist_ok=True)
    if not old.get("identity"):
        old.update({"identity": identity, "sections": {}})
        old["snapshot_files"] = _snapshot(output, identity)
    return output, old


def _finish(output, manifest, section, files, summary):
    # A section owns only its registered files. Remove obsolete generated files
    # after replacing the section, never other notebooks' results or input files.
    previous = manifest["sections"].get(section, {}).get("files", {})
    for relative in set(previous) - set(files):
        path = (output / relative).resolve()
        if path.is_relative_to(output) and path.is_file():
            path.unlink()
    manifest["sections"][section] = {
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "files": {relative: _hash(output / relative) for relative in files},
        "summary": summary,
    }
    if set(SECTION_NAMES).issubset(manifest["sections"]):
        # Once every section is current, the earlier rerun notice is resolved.
        manifest.pop("notice", None)
        manifest.pop("previous_results", None)
    if input_identity() != manifest["identity"]:
        (output / "summary.md").write_text(
            "# Results need to be rerun\n\nInputs changed while saving. These files are not a valid current run.\n",
            encoding="utf-8",
        )
        raise RuntimeError("Inputs changed while outputs were being saved. Rerun before using these files.")
    _write_json(output / "run_manifest.json", manifest)
    lines = [
        "# Current results", "",
        "These are design calculations. A passing screen does not close a requirement or qualify hardware.", "",
        "[Inputs](inputs_used/inputs.csv) · [Choices](inputs_used/choices.csv) · [Source settings](inputs_used/) · [Plots](plots/) · [Result tables](tables/) · [Run details](run_manifest.json)", "",
    ]
    if manifest.get("notice"):
        lines += [f"**Previous results invalidated:** {manifest['notice']}", ""]
    for key, label in SECTION_NAMES.items():
        lines += [f"## {label}", ""]
        if key in manifest["sections"]:
            saved = manifest["sections"][key]
            lines += [f"Saved {saved['saved_utc']}", "", saved["summary"], ""]
            plots = [name for name in saved["files"] if name.endswith(".png")]
            if plots:
                lines += ["Plots: " + " · ".join(
                    f"[{Path(name).stem.removeprefix(key + '_').replace('_', ' ')}]({name})"
                    for name in plots
                ), ""]
        else:
            lines += ["**Not run for these inputs.** Run the corresponding notebook to add this section.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def _save_tables(output, section, tables):
    paths = []
    for name, table in (tables or {}).items():
        relative = f"tables/{section}_{_name(name)}.csv"
        pd.DataFrame(table).to_csv(output / relative, index=False)
        paths.append(relative)
    return paths


def _save_figures(output, section, figures):
    paths = []
    if figures is None:
        return paths
    if not isinstance(figures, dict):
        figures = {f"figure_{number:02d}": figure for number, figure in enumerate(figures, 1)}
    for name, figure in figures.items():
        relative = f"plots/{section}_{_name(name)}.png"
        figure.savefig(output / relative, dpi=140, bbox_inches="tight")
        paths.append(relative)
    return paths


def save_flight(context, departure=None, figures=None, output_dir=None, tables=None, expected_inputs=None):
    """Save the current ascent/departure and its plots; return the output directory."""
    from .flight_evidence import flight_summary
    from .workspace import fingerprint

    history, summary, simulation = context
    if simulation.c != load_config():
        raise ValueError("Flight configuration differs from saved inputs. Save those settings and rerun first.")
    if getattr(simulation, "_flight_fingerprint", fingerprint()) != fingerprint():
        raise ValueError("Flight code or inputs changed after this ascent. Rerun before saving.")
    output, manifest = _prepare(output_dir, expected_inputs)
    metrics = flight_summary(context)
    result_rows = []
    for label, key, unit, limit in [
        ("AoA at max-Qalpha", "alpha_at_max_qalpha_deg", "deg", load_study("flight_review")["max_qalpha_aoa_deg"]),
        ("Maximum axial acceleration", "max_axial_g", "g", simulation.c["limits"]["axial_g"]),
        ("Maximum lateral acceleration", "max_lateral_g", "g", simulation.c["limits"]["lateral_g"]),
    ]:
        value = metrics[key]
        # The user's AoA request is strictly below its stated threshold.
        passed = value < limit if key == "alpha_at_max_qalpha_deg" else value <= limit
        result_rows.append({"Quantity": label, "Value": value, "Unit": unit, "Limit": limit, "Result": "SCREEN_PASS" if passed else "SCREEN_FAIL"})
    for label, key, unit in [
        ("Maximum dynamic pressure", "max_q_Pa", "Pa"),
        ("Maximum Qalpha", "max_qalpha_Pa_deg", "Pa deg"),
        ("Steering loss", "steering_loss_m_s", "m/s"),
    ]:
        result_rows.append({"Quantity": label, "Value": metrics[key], "Unit": unit, "Limit": "—", "Result": "CALCULATED"})
    for label, key in [
        ("Minimum powered control margin", "minimum_powered_control_ratio_margin_Nm"),
        ("Minimum all-phase control margin", "minimum_control_ratio_margin_Nm"),
    ]:
        result_rows.append({"Quantity": label, "Value": metrics[key], "Unit": "N m", "Limit": ">= 0", "Result": "SCREEN_PASS" if metrics[key] >= 0 else "SCREEN_FAIL"})
    results = pd.DataFrame(result_rows)
    flight_tables = {
        "history": history,
        "key_results": results,
        "load_cases": simulation.load_cases,
        "events": simulation.events,
        "metrics": [{"quantity": key, "value": value} for key, value in metrics.items() if not isinstance(value, (dict, list))],
        **(tables or {}),
    }
    files = _save_tables(output, "flight", flight_tables)
    _write_json(output / "tables/flight_summary.json", summary)
    files.append("tables/flight_summary.json")
    text = _markdown(results) + "\n\n[Flight history](tables/flight_history.csv) · [Correlated load cases](tables/flight_load_cases.csv) · [Events](tables/flight_events.csv)\n\nControl results depend on the saved actuator, aerodynamic and geometry assumptions. Coast and roll authority must be supported separately."
    if departure is not None:
        compact = dict(departure)
        departure_history = compact.pop("history", [])
        files += _save_tables(output, "flight", {
            "departure_history": departure_history,
            "departure_results": [{"quantity": key, "value": value} for key, value in compact.items() if not isinstance(value, (dict, list))],
        })
        _write_json(output / "tables/flight_departure.json", compact)
        files.append("tables/flight_departure.json")
        text += f"\n\nDeparture ended with **{departure['status']}**. "
        if "achieved_C3_km2_s2" in departure:
            text += f"C3 = {departure['achieved_C3_km2_s2']:.3f} km²/s²; target = {departure['target_C3_km2_s2']:.3f} km²/s². "
        text += "This energy screen does not establish a planetary encounter."
    else:
        text += "\n\n**Departure has not been calculated for this saved flight.**"
    if figures is None:
        from .plotting import plot_results
        import matplotlib.pyplot as plt
        generated = plot_results(history, summary, simulation.c)
        files += _save_figures(output, "flight", generated)
        for figure in generated:
            plt.close(figure)
    else:
        files += _save_figures(output, "flight", figures)
    return _finish(output, manifest, "flight", files, text)


def save_engineering(tables, figures=None, output_dir=None, expected_inputs=None):
    """Save named engineering tables and figures; return the output directory."""
    output, manifest = _prepare(output_dir, expected_inputs)
    files = _save_tables(output, "engineering", tables)
    files += _save_figures(output, "engineering", figures)
    summary = "Saved studies:\n\n" + "\n".join(
        f"- [{name.replace('_', ' ')}](tables/engineering_{_name(name)}.csv) — {len(table)} rows"
        for name, table in tables.items()
    )
    summary += "\n\nStudy results use the assumptions displayed in the notebook and saved inputs. Missing inputs and failed checks remain open in Verification."
    return _finish(output, manifest, "engineering", files, summary)


def save_verification(results, details, output_dir=None, expected_inputs=None):
    """Save the assigned analyses, their coverage and results without claiming full verification."""
    output, manifest = _prepare(output_dir, expected_inputs)
    tables = output / "tables"
    index = activity_index()
    contracts = input_contracts()
    results = results[results.activity_id.isin(index.activity_id)].copy()
    details = {key: value for key, value in details.items() if key in set(results.activity_id)}
    export(results, details, tables)
    index.to_csv(tables / "activities.csv", index=False)
    contracts.to_csv(tables / "calculation_inputs.csv", index=False)
    files = [f"tables/{name}" for name in (
        "activity_results.csv", "requirement_status.csv", "calculation_results.json",
        "activities.csv", "calculation_inputs.csv",
    )]
    compact = json.loads((tables / "calculation_results.json").read_text(encoding="utf-8"))
    files += sorted({f"tables/{result['history_csv']}" for result in compact.values() if "history_csv" in result})
    counts = results.status.value_counts().rename_axis("Activity status").reset_index(name="Count")
    failing = results[results.status.isin(["SCREEN_FAIL", "REVIEWED_FAIL", "INVALID_INPUTS", "INVALID_EVIDENCE"])]
    summary = (
        f"**{len(index)} assigned activities · {len(contracts)} calculation routes.** "
        f"Results saved for {results.activity_id.nunique()} of those activities. "
        "This is Hyperion's assigned numerical contribution; the complete shared-drive workbook remains authoritative.\n\n"
        "[Assignment manifest](inputs_used/verification.toml) · "
        "[Activity coverage and code routes](tables/activities.csv) · "
        "[Results](tables/activity_results.csv) · "
        "[Named follow-up work](inputs_used/verification_gaps.csv)\n\n"
        + _markdown(counts)
    )
    if not failing.empty:
        checks = load_checks()
        failure_rows = []
        for row in failing.to_dict("records"):
            measurements = []
            specification = checks.get(row["activity_id"], checks.get(row.get("kernel"), {}))
            for rule in specification.get("acceptance", []):
                try:
                    if acceptance(details[row["activity_id"]], [rule]):
                        continue
                    value = details[row["activity_id"]]
                    for key in rule["metric"].split("."):
                        value = value[key]
                    measurements.append(f"{rule['metric']}: {value:.6g} {rule['units']}; required {rule['operator']} {rule['limit']:.6g}")
                except (KeyError, TypeError, ValueError):
                    continue
            failure_rows.append({
                "Activity": row["activity_id"], "Status": row["status"],
                "Result": "; ".join(measurements) or row["detail"],
            })
        summary += "\n\nFailed or invalid assigned activities:\n\n" + _markdown(pd.DataFrame(failure_rows))
    coverage = index.groupby("artifact", sort=False).agg(
        Activities=("activity_id", "count"),
        Coverage=("coverage", lambda values: " ".join(dict.fromkeys(values))),
    ).rename_axis("Artifact").reset_index()
    summary += "\n\nAssigned analysis coverage:\n\n" + _markdown(coverage)
    summary += (
        "\n\n[Parent-requirement coverage](tables/requirement_status.csv) records assigned coverage separately from all source activities. "
        "SCREEN_PASS means the stated calculation criterion passed. Only complete, current, reviewed evidence for every source activity, "
        "with open parameters resolved, can produce REVIEWED_COMPLETE. Missing or stale evidence never passes."
    )
    return _finish(output, manifest, "verification", files, summary)
