"""Editable inputs must agree across flight, studies, choices and verification."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.inputs import (
    INPUTS, _resolve, choices_table, input_tables, load_checks, load_config,
    load_mission_cases, load_study,
)


class SharedInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.inputs = Path(self.temporary.name)
        (self.inputs / "requirements").mkdir()
        for path in INPUTS.glob("*.toml"):
            shutil.copy2(path, self.inputs / path.name)
        shutil.copy2(INPUTS / "requirements/mission_cases.json",
                     self.inputs / "requirements/mission_cases.json")

    def tearDown(self):
        self.temporary.cleanup()

    def replace_setting(self, filename, old, new):
        path = self.inputs / filename
        content = path.read_text(encoding="utf-8")
        self.assertIn(old, content)
        path.write_text(content.replace(old, new), encoding="utf-8")

    def test_hardware_changes_reach_studies_and_verification(self):
        current = load_config(self.inputs)
        diameter = current["vehicle"]["diameter_m"]
        startup = current["vehicle"]["stages"][1]["startup_propellant_kg"]
        self.replace_setting("vehicle.toml", f"diameter_m = {diameter}",
                             "diameter_m = 6.2")
        self.replace_setting("vehicle.toml", f"startup_propellant_kg = {startup}",
                             "startup_propellant_kg = 123.0")
        self.assertEqual(load_study("tank_thickness", self.inputs)["inputs"]["radius_m"], 3.1)
        checks = load_checks(self.inputs)
        self.assertEqual(checks["integrated_tmi"]["inputs"]["restart_propellant_kg"], 123.0)
        flight = load_config(self.inputs)
        sizing = checks["sourced_sizing"]["inputs"]
        self.assertEqual(sizing["tanks"][0]["loaded_mass_kg"],
                         flight["vehicle"]["stages"][0]["tanks"][1]["loaded_mass_kg"])
        self.assertEqual(checks["PROP.3.TANK.02.A01"]["inputs"], sizing)
        self.assertEqual(checks["PERF.2.32.A02"]["inputs"], sizing)

    def test_choices_display_actual_changed_values(self):
        selected = load_config(self.inputs)["vehicle"]["fairing_length_m"]
        self.replace_setting("vehicle.toml", f"fairing_length_m = {selected}",
                             "fairing_length_m = 12.25")
        choices = choices_table(self.inputs).set_index("Choice")
        self.assertIn("12.25 m", choices.loc["Vehicle geometry", "Selection"])
        self.assertNotIn(str(selected), choices.loc["Vehicle geometry", "Selection"])

    def test_inconsistent_stage_and_tank_propellant_is_rejected(self):
        stage = load_config(self.inputs)["vehicle"]["stages"][0]
        total = stage["propellant_kg"]
        self.replace_setting("vehicle.toml", f"propellant_kg = {total}",
                             f"propellant_kg = {total + 100}")
        with self.assertRaisesRegex(ValueError, "Stage 1: propellant_kg.*Update both"):
            load_config(self.inputs)

    def test_propellant_source_roundoff_is_allowed(self):
        stage = load_config(self.inputs)["vehicle"]["stages"][0]
        total = stage["propellant_kg"]
        self.replace_setting("vehicle.toml", f"propellant_kg = {total}",
                             f"propellant_kg = {total + 1e-7}")
        self.assertEqual(load_config(self.inputs)["vehicle"]["stages"][0]["propellant_kg"],
                         total + 1e-7)

    def test_installed_thrust_checks_follow_both_selected_stages(self):
        config = load_config(self.inputs)
        for stage, selected_thrust in zip(config["vehicle"]["stages"], (7200000, 990000)):
            self.replace_setting("vehicle.toml", f"thrust_N = {stage['thrust_N']}",
                                 f"thrust_N = {selected_thrust}")
        checks = load_checks(self.inputs)
        for activity, expected in (("PROP.3.SYS.04.A02", 7200000),
                                   ("PROP.3.SYS.05.A02", 990000)):
            inputs = checks[activity]["inputs"]
            self.assertEqual(inputs["per_engine_thrust_N"] * inputs["engine_count"], expected)

    def test_display_covers_every_flight_setting(self):
        def leaves(value, path=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from leaves(item, f"{path}.{key}" if path else key)
            elif isinstance(value, list) and any(isinstance(item, dict) for item in value):
                for index, item in enumerate(value):
                    yield from leaves(item, f"{path}.{index}")
            else:
                yield path

        tables = input_tables(self.inputs)
        displayed = {key for table in tables.values() for key in table["Input"]}
        self.assertTrue(set(leaves(load_config(self.inputs))).issubset(displayed))
        for table in tables.values():
            self.assertTrue(table["Source"].str.contains(".toml", regex=False).all())
            self.assertFalse(any(isinstance(value, (dict, list)) for value in table["Value"]))
        self.assertIn("activities.PERF.2.06.A01.acceptance.0.limit", displayed)
        self.assertIn("studies.tank_thickness.inputs.radius_m", displayed)

    def test_mission_vectors_are_reused_and_studies_are_not_evidence(self):
        checks = load_checks(self.inputs)
        mission = load_mission_cases(self.inputs)
        self.assertEqual(checks["lambert_window"]["inputs"]["cases"], mission["cases"])
        self.assertNotIn("studies", checks)
        self.assertNotIn("tank_thickness", checks)
        self.assertFalse(any("example" in key for key in checks))
        targets = load_study("departure_targets", self.inputs)["targets"]
        self.assertEqual(targets[1]["target_c3_km2_s2"],
                         checks["integrated_tmi"]["inputs"]["target_c3_km2_s2"])

    def test_returns_independent_dictionaries(self):
        checks = load_checks(self.inputs)
        original = checks["sourced_sizing"]["inputs"]["tanks"][0]["loaded_mass_kg"]
        checks["PERF.2.32.A02"]["inputs"]["tanks"][0]["loaded_mass_kg"] = -1
        self.assertEqual(checks["sourced_sizing"]["inputs"]["tanks"][0]["loaded_mass_kg"], original)
        self.assertEqual(load_checks(self.inputs)["PERF.2.32.A02"]["inputs"]["tanks"][0]["loaded_mass_kg"], original)

    def test_explicit_json_records_still_load(self):
        record = self.inputs / "saved_run.json"
        record.write_text(json.dumps({"custom": {"input": 42}}), encoding="utf-8")
        self.assertEqual(load_config(record), {"custom": {"input": 42}})
        self.assertEqual(load_checks(record), {"custom": {"input": 42}})

    def test_invalid_references_fail_before_calculation(self):
        with self.assertRaisesRegex(ValueError, "Circular"):
            _resolve({"ref": "a"}, {"a": {"ref": "b"}, "b": {"ref": "a"}})
        with self.assertRaisesRegex(ValueError, "Unknown input reference"):
            _resolve({"ref": "unknown"}, {})
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            _resolve({"ref": "name", "scale": 2}, {"name": "engine"})
        with self.assertRaisesRegex(ValueError, "only contain ref and scale"):
            _resolve({"ref": "x", "expression": "x + 1"}, {"x": 2})


if __name__ == "__main__":
    unittest.main()
