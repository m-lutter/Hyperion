"""Keep current results traceable when settings, evidence or cached files change."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from model import outputs, verify, workspace
from model.tools import build_verification_index
from model.verification import activity_index, baseline_id


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "current"
        self.identity = {"input_id": "first", "baseline_id": "model-a", "files": {}}
        self.patches = [
            patch.object(outputs, "input_identity", side_effect=lambda: self.identity.copy()),
            patch.object(outputs, "input_tables", return_value={"vehicle": pd.DataFrame({"mass_kg": [10]})}),
            patch.object(outputs, "choices_table", return_value=pd.DataFrame({"choice": ["assumed control"]})),
            patch.object(outputs, "load_config", return_value={"mass_kg": 10}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def save_study(self):
        return outputs.save_engineering(
            {"tank": pd.DataFrame({"ullage_fraction": [0.016]})},
            output_dir=self.folder,
            expected_inputs=self.identity.copy(),
        )

    def test_changed_inputs_start_clean_current_results(self):
        self.save_study()
        self.identity = {"input_id": "second", "baseline_id": "model-b", "files": {}}
        with self.assertWarns(UserWarning):
            outputs.save_engineering({"new": pd.DataFrame({"value": [2]})}, output_dir=self.folder)
        self.assertFalse((self.folder / "tables/engineering_tank.csv").exists())
        manifest = json.loads((self.folder / "run_manifest.json").read_text())
        archive = Path(manifest["previous_results"])
        self.assertTrue((archive / "tables/engineering_tank.csv").exists())
        summary = (self.folder / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Previous results invalidated", summary)
        self.assertIn("Not run for these inputs", summary)
        self.assertEqual(manifest["identity"], self.identity)

    def test_changed_inputs_during_calculation_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "changed after"):
            outputs.save_engineering({}, output_dir=self.folder, expected_inputs={"input_id": "old"})
        self.assertFalse(self.folder.exists())

    def test_modified_saved_table_is_not_reused(self):
        self.save_study()
        (self.folder / "tables/engineering_tank.csv").write_text("edited output")
        with self.assertWarns(UserWarning):
            self.save_study()
        manifest = json.loads((self.folder / "run_manifest.json").read_text())
        self.assertIn("changed or was missing", manifest["notice"])

    def test_modified_source_snapshot_is_invalidated_and_restored(self):
        root = Path(self.temp.name) / "repo"
        source = root / "inputs/vehicle.toml"
        source.parent.mkdir(parents=True)
        source.write_text("mass_kg = 10")
        self.identity["files"] = {"inputs/vehicle.toml": hashlib.sha256(source.read_bytes()).hexdigest()}
        with patch.object(outputs, "ROOT", root):
            self.save_study()
            snapshot = self.folder / "inputs_used/vehicle.toml"
            snapshot.write_text("mass_kg = 999")
            with self.assertWarns(UserWarning):
                self.save_study()
        self.assertEqual(snapshot.read_bytes(), source.read_bytes())
        manifest = json.loads((self.folder / "run_manifest.json").read_text())
        self.assertIn("input snapshot changed", manifest["notice"])

    def test_selected_aero_parent_path_is_copied_and_missing_copy_invalidates(self):
        root = Path(self.temp.name) / "repo"
        (root / "inputs").mkdir(parents=True)
        aero = root / "selected_aero.csv"
        aero.write_text("Mach,AoA,CD\n1,0,0.2\n")
        self.patches[0].stop()
        config = {"aero": {"file": "inputs/../selected_aero.csv"}}
        with patch.object(outputs, "ROOT", root), patch.object(outputs, "load_config", return_value=config), patch.object(outputs, "baseline_id", return_value="test-model"):
            identity = outputs.input_identity()
            self.assertEqual(identity["selected_aerodynamics"]["path"], "selected_aero.csv")
            outputs.save_engineering({}, output_dir=self.folder, expected_inputs=identity)
            snapshot = self.folder / "inputs_used/selected_aerodynamics.csv"
            self.assertEqual(snapshot.read_bytes(), aero.read_bytes())
            snapshot.unlink()
            with self.assertWarns(UserWarning):
                outputs.save_engineering({}, output_dir=self.folder, expected_inputs=identity)
            self.assertEqual(snapshot.read_bytes(), aero.read_bytes())

    def test_same_baseline_retains_sections_and_exports_honest_status(self):
        self.save_study()
        requirement = {"id": "T.01", "domain": "test", "activities": ["calculate"], "open_parameters": [], "source_ref": "test!A1"}
        results = pd.DataFrame([{"activity_id": "T.01.A01", "requirement_id": "T.01", "status": "SCREEN_PASS", "detail": "criterion passed", "kernel": "test", "artifact": "Assigned analysis", "coverage": "Numerical screen only"}])
        with patch("model.verification.load_requirements", return_value=[requirement]), patch("model.verification.activities", return_value=results), patch.object(outputs, "activity_index", return_value=results), patch.object(outputs, "input_contracts", return_value=pd.DataFrame({"calculation": ["test"]})):
            outputs.save_verification(results, {"T.01.A01": {"value": 1}}, output_dir=self.folder, expected_inputs=self.identity)
        manifest = json.loads((self.folder / "run_manifest.json").read_text())
        self.assertEqual(set(manifest["sections"]), {"engineering", "verification"})
        saved = pd.read_csv(self.folder / "tables/requirement_status.csv")
        self.assertEqual(saved.status.tolist(), ["OPEN"])
        self.assertTrue((self.folder / "inputs_used/choices.csv").exists())
        self.assertTrue((self.folder / "tables/activities.csv").exists())
        self.assertIn("Only complete, current, reviewed evidence", (self.folder / "summary.md").read_text(encoding="utf-8"))

    def test_verification_summary_limits_claims_to_assignments_and_shows_failed_values(self):
        requirement = {"id": "T.01", "domain": "test", "activities": ["calculate", "inspect"], "open_parameters": [], "source_ref": "test!A1"}
        assigned = pd.DataFrame([{"activity_id": "T.01.A01", "requirement_id": "T.01", "status": "SCREEN_FAIL", "detail": "numerical limit", "kernel": "test", "artifact": "Load report", "coverage": "Rigid-body acceleration only"}])
        unrelated = assigned.assign(activity_id="OTHER.A01", requirement_id="OTHER", detail="outside this assignment")
        results = pd.concat([assigned, unrelated], ignore_index=True)
        details = {"T.01.A01": {"load_g": 7}, "OTHER.A01": {"load_g": 99}}
        checks = {"test": {"acceptance": [{"metric": "load_g", "operator": "<=", "limit": 6, "units": "g", "source": "T.01"}]}}
        with patch("model.verification.load_requirements", return_value=[requirement]), patch("model.verification.activities", return_value=assigned), patch.object(outputs, "activity_index", return_value=assigned), patch.object(outputs, "input_contracts", return_value=pd.DataFrame({"calculation": ["test"]})), patch.object(outputs, "load_checks", return_value=checks):
            outputs.save_verification(results, details, output_dir=self.folder, expected_inputs=self.identity)
        summary = (self.folder / "summary.md").read_text(encoding="utf-8")
        self.assertIn("1 assigned activities", summary)
        self.assertIn("load_g: 7 g; required <= 6", summary)
        self.assertIn("Rigid-body acceleration only", summary)
        self.assertIn("inputs_used/verification.toml", summary)
        self.assertIn("inputs_used/verification_gaps.csv", summary)
        self.assertNotIn("OTHER.A01", summary)
        saved = json.loads((self.folder / "tables/calculation_results.json").read_text())
        self.assertEqual(set(saved), {"T.01.A01"})
        coverage = pd.read_csv(self.folder / "tables/requirement_status.csv")
        self.assertEqual(coverage.assigned_activities.tolist(), [1])
        self.assertEqual(coverage.expected_activities.tolist(), [2])
        self.assertFalse(coverage.complete_activity_set.iloc[0])

    def test_full_identity_includes_evidence_without_circular_baseline(self):
        # The evidence record may contain its reviewed baseline ID, but that
        # record's own hash must not change the baseline it refers to.
        root = Path(self.temp.name) / "repo"
        for path in ("model/inputs.py", "inputs/vehicle.toml", "inputs/flight.toml", "inputs/checks.toml", "inputs/aerodynamics.csv", "inputs/requirements/requirements.json", "inputs/requirements/mission_cases.json", "inputs/evidence/records.json"):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("initial")
        with patch("model.verification.ROOT", root), patch.object(outputs, "ROOT", root), patch("model.verification.load_config", return_value={"aero": {"file": "inputs/aerodynamics.csv"}}), patch.object(outputs, "load_config", return_value={"aero": {"file": "inputs/aerodynamics.csv"}}):
            first_baseline = baseline_id()
            # Stop the setUp identity patch for this source-file test.
            self.patches[0].stop()
            first_inputs = outputs.input_identity()
            (root / "inputs/evidence/records.json").write_text("reviewed record")
            self.assertEqual(first_baseline, baseline_id())
            self.assertNotEqual(first_inputs, outputs.input_identity())
            (root / "inputs/requirements/mission_cases.json").write_text("new state vectors")
            self.assertNotEqual(first_baseline, baseline_id())


class CacheTests(unittest.TestCase):
    def test_cache_invalidates_for_changed_inputs_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = {"inputs/flight.toml": "first"}
            context = (pd.DataFrame({"t_s": [0, 1]}), {"events": []}, SimpleNamespace(c={}))
            with patch.object(workspace, "ROOT", root), patch.object(workspace, "fingerprint", side_effect=lambda: identity.copy()), patch.object(workspace, "run_trajectory", return_value=context) as run, patch.object(workspace, "Simulation", side_effect=lambda config: SimpleNamespace(c=config)), patch.object(workspace, "load_config", return_value={}), patch.object(workspace, "summarize", return_value=({}, [])):
                workspace.current_flight()
                workspace.current_flight()
                self.assertEqual(run.call_count, 1)
                identity["inputs/flight.toml"] = "changed"
                workspace.current_flight()
                self.assertEqual(run.call_count, 2)
                (root / ".cache/flight/trajectory.csv").write_text("changed")
                workspace.current_flight()
                self.assertEqual(run.call_count, 3)

    def test_inputs_changed_during_integration_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            context = (pd.DataFrame({"t_s": [0]}), {"events": []}, SimpleNamespace(c={}))
            with patch.object(workspace, "ROOT", Path(temp)), patch.object(workspace, "fingerprint", side_effect=[{"x": "before"}, {"x": "after"}]), patch.object(workspace, "run_trajectory", return_value=context):
                with self.assertRaisesRegex(RuntimeError, "changed during"):
                    workspace.current_flight()
            self.assertFalse((Path(temp) / ".cache/flight/fingerprint.json").exists())


class CommandLineTests(unittest.TestCase):
    def test_activity_index_records_the_shared_dispatch_routes(self):
        table = pd.DataFrame({"kernel": ["calculate", None]})
        function = SimpleNamespace(__module__="model.example", __name__="calculate")
        with patch("model.verification.activities", return_value=table), patch("model.verification.catalog", return_value=({"calculate": function}, {})):
            index = activity_index()
        self.assertEqual(index.code_route.iloc[0], "model.example.calculate")
        self.assertEqual(index.route_type.tolist(), ["calculation", "external evidence validation"])

    def test_index_utility_never_changes_current_run_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = root / "outputs/current/tables/activities.csv"
            saved.parent.mkdir(parents=True)
            saved.write_text("saved run bytes")
            index = pd.DataFrame({"kernel": ["calculate"], "code_route": ["model.example.calculate"]})
            contracts = pd.DataFrame({"calculation": ["calculate"]})
            with patch.object(build_verification_index, "ROOT", root), patch.object(build_verification_index, "activity_index", return_value=index), patch.object(build_verification_index, "input_contracts", return_value=contracts), patch("builtins.print"):
                build_verification_index.main()
                self.assertFalse((root / ".cache").exists())
                build_verification_index.main(write_cache=True)
            self.assertEqual(saved.read_text(), "saved run bytes")
            cached = pd.read_csv(root / ".cache/verification_index/activities.csv")
            pd.testing.assert_frame_equal(index, cached)

    def test_selected_departure_ignores_reviewed_records_and_activity_overrides(self):
        settings = {"coast_duration_s": 600, "target_c3_km2_s2": 9.8}
        selected = {"status": "selected departure"}
        for details in ({}, {"T.A01": {"status": "overridden departure"}}):
            with self.subTest(details=details), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                (output / "run_manifest.json").write_text("{}")
                context = (None, {"events": [{"event": "orbital_speed_cutoff"}]}, None)
                with patch.object(verify, "input_identity", return_value={}), patch.object(verify, "current_flight", return_value=context) as ascent, patch.object(verify, "load_checks", return_value={"integrated_tmi": {"inputs": settings}}), patch.object(verify, "integrated_tmi", return_value=selected) as departure, patch.object(verify, "evaluate", return_value=(pd.DataFrame(), details)), patch.object(verify, "save_flight") as save, patch.object(verify, "save_verification", return_value=output):
                    verify.run(output)
                ascent.assert_called_once_with(recompute=False)
                departure.assert_called_once_with(context=context, **settings)
                self.assertIs(save.call_args.kwargs["departure"], selected)

    def test_no_orbital_cutoff_skips_departure_but_saves_ascent(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "run_manifest.json").write_text("{}")
            context = (None, {"events": []}, None)
            with patch.object(verify, "input_identity", return_value={}), patch.object(verify, "current_flight", return_value=context), patch.object(verify, "integrated_tmi") as departure, patch.object(verify, "evaluate", return_value=(pd.DataFrame(), {})), patch.object(verify, "save_flight") as save, patch.object(verify, "save_verification", return_value=output):
                verify.run(output)
            departure.assert_not_called()
            self.assertIsNone(save.call_args.kwargs["departure"])


if __name__ == "__main__":
    unittest.main()
