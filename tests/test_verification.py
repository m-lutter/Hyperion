import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from model.verification import (
    acceptance,
    activities,
    catalog,
    check_evidence,
    evaluate,
    export,
    input_contracts,
    load_scope,
    measured_limits,
    requirement_status,
)


class EvidenceTests(unittest.TestCase):
    def test_missing_is_not_zero_or_pass(self):
        self.assertEqual(check_evidence({})[0], "EVIDENCE_REQUIRED")
        with self.assertRaises(ValueError):
            acceptance(
                {"x": float("nan")},
                [
                    {
                        "metric": "x",
                        "operator": "<=",
                        "limit": 2,
                        "units": "m",
                        "source": "test",
                    }
                ],
            )

    def test_reviewed_file_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence_file = Path(temp) / "record.txt"
            evidence_file.write_text("measured evidence")
            record = dict(
                file="record.txt",
                sha256=hashlib.sha256(evidence_file.read_bytes()).hexdigest(),
                reviewer="Example reviewer",
                date="2026-10-28",
                result="pass",
                configuration_id="test only",
                acceptance_criteria="x<=1 m",
                scope="test fixture",
            )
            self.assertEqual(check_evidence(record, temp)[0], "REVIEWED_PASS")
            self.assertEqual(
                check_evidence(record, temp, expected_configuration="new-baseline")[0],
                "STALE_EVIDENCE",
            )
            evidence_file.write_text("changed")
            self.assertEqual(check_evidence(record, temp)[0], "INVALID_EVIDENCE")

    def test_boundary_and_source(self):
        rule = {
            "metric": "g",
            "operator": "<=",
            "limit": 6,
            "units": "g",
            "source": "test",
        }
        self.assertTrue(acceptance({"g": 6}, [rule]))
        self.assertFalse(acceptance({"g": 6.001}, [rule]))
        self.assertIsNone(acceptance({"g": 1}, []))
        rule.pop("source")
        with self.assertRaises(ValueError):
            acceptance({"g": 1}, [rule])

    def test_measurement_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            measurements = Path(temp) / "measurements.csv"
            header = (
                "case_id,quantity,value,unit,lower_limit,upper_limit,criterion_source\n"
            )
            measurements.write_text(
                header
                + "A,force,10,N,0,10,requirement\nB,force,11,N,0,10,requirement\n"
            )
            frame = measured_limits(measurements)
            self.assertEqual(frame.within_limits.tolist(), [True, False])
            measurements.write_text(header + "A,force,,N,0,10,requirement\n")
            with self.assertRaises(ValueError):
                measured_limits(measurements)

    def test_partial_review_cannot_close_requirement(self):
        requirement = dict(
            id="TEST.1.01",
            domain="test",
            activities=["a", "b"],
            open_parameters=[],
            source_ref="test!A1",
        )
        rows = pd.DataFrame(
            [
                dict(
                    activity_id="TEST.1.01.A01",
                    requirement_id="TEST.1.01",
                    status="REVIEWED_PASS",
                )
            ]
        )
        with patch(
            "model.verification.load_requirements", return_value=[requirement]
        ), patch("model.verification.activities", return_value=rows):
            status = requirement_status(rows).iloc[0]
            self.assertEqual(status.status, "OPEN")
            self.assertTrue(status.assigned_reviewed_complete)
            self.assertFalse(status.complete_activity_set)
            self.assertEqual(status.expected_activities, 2)
            self.assertEqual(status.assigned_activities, 1)
            duplicate = pd.concat([rows, rows])
            self.assertEqual(requirement_status(duplicate).iloc[0].status, "OPEN")

    def test_every_activity_has_valid_dispatch(self):
        functions, assignments = catalog()
        self.assertEqual(assignments, {})
        self.assertEqual(set(functions), {"flight_summary", "integrated_tmi", "lambert_window", "sourced_sizing", "thrust_sum"})
        with tempfile.TemporaryDirectory() as temp:
            empty_file = Path(temp) / "empty.json"
            empty_file.write_text("{}")
            result, _ = evaluate(
                input_file=empty_file, evidence_file=empty_file, context=None
            )
            self.assertEqual(set(result.activity_id), set(activities().activity_id))
            self.assertFalse(result.status.str.contains("PASS").any())


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.requirements = [{
            "id": "TEST.1.01", "domain": "test", "statement": "Test requirement",
            "activities": ["Assigned first.", "Assigned second.", "Unassigned activity."],
            "methods": ["Analysis"], "source_ref": "test!G5", "open_parameters": [],
            "source_file": "test.xlsx", "source_sha256": "source digest",
        }]

    def entry(self, number=1, **changes):
        entry = {
            "activity_id": f"TEST.1.01.A{number:02d}",
            "calculation": "thrust_sum",
            "expected_activity": self.requirements[0]["activities"][number - 1],
            "artifact": "Installed thrust", "coverage": "Arithmetic only.",
        }
        return {**entry, **changes}

    def scope(self, entries):
        path = self.folder / "verification.toml"
        lines = ['description = "Test assigned scope"']
        for entry in entries:
            lines.append("\n[[activities]]")
            lines.extend(f"{key} = {json.dumps(value)}" for key, value in entry.items())
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_default_scope_is_exactly_the_nineteen_assigned_activities(self):
        expected = {
            "MIS.0.01.A01", "MIS.0.01.A02", "SYS.1.02.A01", "SYS.1.08.A01",
            "SYS.1.09.A02", "SYS.1.16.A01", "PERF.2.05.A01", "PERF.2.06.A01",
            "PERF.2.13.A02", "PERF.2.32.A02", "PERF.2.35.A02",
            "PROP.3.SYS.03.A01", "PROP.3.SYS.04.A02", "PROP.3.SYS.05.A02",
            "PROP.3.SYS.16.A01", "PROP.3.TANK.01.A01", "PROP.3.TANK.02.A01",
            "PROP.3.STG1.01.A01", "PROP.3.STG2.01.A01",
        }
        table = activities()
        self.assertEqual(set(table.activity_id), expected)
        self.assertEqual(len(table), len(expected))
        self.assertTrue(table[["artifact", "coverage"]].notna().all().all())
        self.assertEqual(set(table.kernel), set(catalog()[0]))

    def test_scope_rejects_duplicate_unknown_route_unknown_id_and_stale_wording(self):
        invalid = [
            ([self.entry(), self.entry()], "Duplicate scoped activity"),
            ([self.entry(calculation="structures")], "Unknown calculation route"),
            ([self.entry(activity_id="TEST.1.99.A01")], "Unknown scoped activity"),
            ([self.entry(expected_activity="Assigned first")], "wording mismatch"),
        ]
        for entries, message in invalid:
            with self.subTest(message=message), patch("model.verification.load_requirements", return_value=self.requirements):
                with self.assertRaisesRegex(ValueError, message):
                    activities(self.scope(entries))

    def test_invalid_scope_stops_before_loading_inputs_or_running_calculations(self):
        scope = self.scope([self.entry(expected_activity="Old activity wording.")])
        with patch("model.verification.load_requirements", return_value=self.requirements), patch("model.verification.catalog") as registry, patch("model.verification.load_checks") as inputs:
            with self.assertRaisesRegex(ValueError, "wording mismatch"):
                evaluate(scope_file=scope)
        registry.assert_not_called()
        inputs.assert_not_called()

    def test_missing_assigned_result_is_distinct_from_whole_source_coverage(self):
        scope = self.scope([self.entry(), self.entry(2)])
        results = pd.DataFrame([{
            "activity_id": "TEST.1.01.A01", "requirement_id": "TEST.1.01", "status": "REVIEWED_PASS",
        }])
        with patch("model.verification.load_requirements", return_value=self.requirements):
            status = requirement_status(results, scope).iloc[0]
        self.assertEqual(status.assigned_activities, 2)
        self.assertEqual(status.expected_activities, 3)
        self.assertFalse(status.complete_assigned_set)
        self.assertFalse(status.assigned_reviewed_complete)
        self.assertFalse(status.complete_activity_set)

    def test_whole_requirement_closure_requires_reviewed_evidence_and_no_open_parameters(self):
        self.requirements[0]["activities"] = ["Assigned first."]
        scope = self.scope([self.entry()])
        results = pd.DataFrame([{
            "activity_id": "TEST.1.01.A01", "requirement_id": "TEST.1.01", "status": "REVIEWED_PASS",
        }])
        with patch("model.verification.load_requirements", return_value=self.requirements):
            self.assertEqual(requirement_status(results, scope).iloc[0].status, "REVIEWED_COMPLETE")
            results["status"] = "SCREEN_PASS"
            self.assertEqual(requirement_status(results, scope).iloc[0].status, "OPEN")
            results["status"] = "REVIEWED_PASS"
            self.requirements[0]["open_parameters"] = ["TBD-TEST"]
            self.assertEqual(requirement_status(results, scope).iloc[0].status, "OPEN")

    def test_only_assigned_calculations_run_and_shared_results_are_cached(self):
        scope = self.scope([self.entry(), self.entry(2)])
        calls = []

        def thrust_sum(*, per_engine_thrust_N, engine_count):
            calls.append((per_engine_thrust_N, engine_count))
            return {"thrust_N": per_engine_thrust_N * engine_count}

        def unassigned():
            self.fail("An unassigned calculation was invoked.")

        checks = {"thrust_sum": {"inputs": {"per_engine_thrust_N": 8, "engine_count": 2}}}
        with patch("model.verification.load_requirements", return_value=self.requirements), patch("model.verification.catalog", return_value=({"thrust_sum": thrust_sum, "flight_summary": unassigned}, {})), patch("model.verification.load_checks", return_value=checks), patch("model.verification.baseline_id", return_value="test"):
            result, detail = evaluate(evidence_file=self.folder / "absent.json", scope_file=scope)
            contracts = input_contracts(scope)
        self.assertEqual(calls, [(8, 2)])
        self.assertEqual(result.activity_id.tolist(), ["TEST.1.01.A01", "TEST.1.01.A02"])
        self.assertEqual(set(result.status), {"CALCULATED"})
        self.assertEqual(set(detail), set(result.activity_id))
        self.assertEqual(contracts.calculation.tolist(), ["thrust_sum"])

    def test_export_excludes_unassigned_results_and_retains_source_closure_count(self):
        scope = self.scope([self.entry()])
        results = pd.DataFrame([
            {"activity_id": "TEST.1.01.A01", "requirement_id": "TEST.1.01", "status": "REVIEWED_PASS"},
            {"activity_id": "TEST.1.01.A03", "requirement_id": "TEST.1.01", "status": "REVIEWED_PASS"},
            {"activity_id": "OTHER.1.01.A01", "requirement_id": "OTHER.1.01", "status": "OPEN"},
        ])
        detail = {identifier: {"value": 1} for identifier in results.activity_id}
        with patch("model.verification.load_requirements", return_value=self.requirements):
            output = export(results, detail, self.folder / "outputs", scope)
        self.assertEqual(pd.read_csv(output / "activity_results.csv").activity_id.tolist(), ["TEST.1.01.A01"])
        self.assertEqual(set(json.loads((output / "calculation_results.json").read_text())), {"TEST.1.01.A01"})
        status = pd.read_csv(output / "requirement_status.csv")
        self.assertEqual(status.id.tolist(), ["TEST.1.01"])
        self.assertEqual(status.expected_activities.tolist(), [3])
        self.assertEqual(status.assigned_activities.tolist(), [1])
        self.assertEqual(status.status.tolist(), ["OPEN"])


if __name__ == "__main__":
    unittest.main()
