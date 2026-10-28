# Add verification evidence

Use this folder for reviewed tests, inspections, demonstrations, and specialist
analyses. A simulation example is not a hardware test result.

## Add a record

1. Find the activity ID and exact wording in [activities.csv](../../outputs/current/tables/activities.csv).
2. Put the evidence file in this folder. Include its procedure, conditions, results,
   and acceptance limits. For external FEA/CFD, include model assumptions and
   convergence checks.
3. Add an entry keyed by that activity ID to [records.json](records.json), using
   the fields below.
4. Run the [Verification register](../../03_Verification.ipynb) again.

| Field | What to enter |
|---|---|
| `activity_id` | The activity being reviewed |
| `activity_text` | Its exact current wording |
| `file` | Evidence path relative to the repository, such as `inputs/evidence/test-report.pdf` |
| `sha256` | SHA-256 hash of that file |
| `reviewer`, `date` | Reviewer name and date in `YYYY-MM-DD` format |
| `configuration_id` | Current version fingerprint from `baseline_id()` |
| `acceptance_criteria` | Limits, units, and requirement/source references |
| `scope` | Procedure, environment, instrumentation/model validation, and activities covered |
| `result` | The reviewer's decision: `pass` or `fail` |

Get the version fingerprint and file hash from Python in the repository:

```python
from pathlib import Path
from hashlib import sha256
from model.verification import baseline_id

print("Configuration:", baseline_id())
print("File hash:", sha256(Path("inputs/evidence/test-report.pdf").read_bytes()).hexdigest())
```

The checker validates the record, file hash, activity wording, and version. It
cannot judge the quality of the engineering review. Changed evidence becomes
invalid or stale; a passing record covers only its named activity. A requirement
stays open until all its activities pass review and its TBD/TBR values are resolved.

## Check measured values

Copy [measurement_template.csv](measurement_template.csv) and enter one row per
case and quantity. Supply a value, unit, at least one limit, and the limit's source.
Then run:

```python
from model.verification import measured_limits

results = measured_limits("inputs/evidence/measurements.csv")
print(results)
```

This checks each row against its limits. It does not convert units or establish
test coverage. Include the raw data and checked results in the reviewed record.
Keep proprietary or large files in a private checkout; this repository is public.
