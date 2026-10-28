import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from hyperion.verification import (
    acceptance,
    activities,
    catalog,
    check_evidence,
    evaluate,
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
            "hyperion.verification.load_requirements", return_value=[requirement]
        ):
            self.assertEqual(requirement_status(rows).iloc[0].status, "OPEN")
            duplicate = pd.concat([rows, rows])
            self.assertEqual(requirement_status(duplicate).iloc[0].status, "OPEN")

    def test_every_activity_has_valid_dispatch(self):
        functions, assignments = catalog()
        self.assertTrue(set(assignments.values()).issubset(functions))
        with tempfile.TemporaryDirectory() as temp:
            empty_file = Path(temp) / "empty.json"
            empty_file.write_text("{}")
            result, _ = evaluate(
                input_file=empty_file, evidence_file=empty_file, context=None
            )
            self.assertEqual(set(result.activity_id), set(activities().activity_id))
            self.assertFalse(result.status.str.contains("PASS").any())


if __name__ == "__main__":
    unittest.main()
