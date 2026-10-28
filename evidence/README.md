# Evidence records

The calculations produce preliminary design evidence. Hardware tests, inspections,
CAD checks and specialist models need their actual reviewed artifacts. Every
activity has a stable identifier in `data/activities.csv`; no activity disappears
because a model or an acceptance threshold is missing.

Copy evidence files into this folder and add a record keyed by activity ID to
`records.json`. Keep large/proprietary artifacts outside the public repository;
run the same checks in a private local checkout if necessary. Empty records stay
`EVIDENCE_REQUIRED`. Never enter a test pass for a simulated example.

Each record requires:

- `activity_id`: the exact activity being reviewed.
- `file` and `sha256`: repository-relative evidence path and SHA-256 hash.
- `reviewer` and `date`: responsible reviewer and ISO date.
- `configuration_id`: the current content fingerprint returned by
  `hyperion.verification.baseline_id()`; older evidence remains `STALE_EVIDENCE`.
- `activity_text`: exact current activity wording from the register.
- `acceptance_criteria`: the actual limits, units and requirement/source reference.
- `scope`: procedure, environment, instrumentation/model validation and coverage.
- `result`: `pass` or `fail`, determined by the responsible reviewer.

The importer verifies completeness, ID and file integrity. It cannot determine
whether an engineering review or physical test was performed correctly. A reviewed
record only closes its own activity. A requirement with an open TBD/TBR remains open.

Numerical test/analysis results can use `measurement_template.csv`. Call
`hyperion.verification.measured_limits(path)` to calculate every row's result.
It rejects missing values, absent limits and duplicate case/quantity keys. Units,
limits and criterion references must be supplied explicitly; it does not convert
units or infer unprovided test coverage. Include this processed result and the
raw data in the reviewed evidence package.

For external FEA/CFD/combustion/radiation tools, export their numerical results and
mesh/material/boundary-condition/convergence report. The matrix-based finite
element and thermal kernels in this repository can calculate with supplied model
matrices. They do not generate a validated vehicle mesh from CAD automatically.
