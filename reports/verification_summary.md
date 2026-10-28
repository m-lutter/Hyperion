# Verification execution, 28 October 2026

OneDrive was confirmed **Backed up and synced**. The current Verification Plan source hash is recorded in `data/source_manifest.json`.

The register contains **373 requirements / 677 activities**. **109 reusable calculation functions** are available, with **246 activity assignments**. Remaining activities use the external evidence validation route; this is not a claim that all specialist analyses or physical tests have been performed.

| Activity result | Count |
|---|---:|
| EVIDENCE_REQUIRED | 431 |
| MISSING_INPUTS | 227 |
| CALCULATED | 7 |
| SCREEN_FAIL | 6 |
| SCREEN_PASS | 6 |

**No requirement is closed by a numerical screen alone.** All requirements remain open or have failed/invalid evidence until complete reviewed evidence and resolved TBD/TBR values are supplied.

## Governing findings

- The updated ascent has AoA **2.177° at max-Qα**, and its axial/lateral rigid-body acceleration screens pass.
- All-phase controllability fails during engine-off coast; no RCS authority is allocated.
- The 600 s coast/restart prograde TMI screen is propellant limited at **C3 −12.253672 km²/s²**. Both the master 7.7058 and provisional requirement 9.8 targets fail.
- Installed first-stage thrust **6,765 kN** is below provisional **11,000 kN**.
- The source stage1 LOX tank does not provide the provisional **4% ullage** with the source loaded mass and density.
- A 1–6 mm pressure-only tank sweep is recomputed in notebook02. It is a source design trade, not an installed-tank strength result.

## Validation

**94 analytical, adverse-input, evidence-integrity and trajectory regression tests passed.** Five notebooks executed without errors and saved seven inline figures. See `notebook_execution.json` for per-notebook timing and baseline identity.

Tests cover conservation, independent Lambert endpoint propagation, RK45 accounting independent of output spacing, actuator response, FE/modal/thermal analytical cases, source-state rejection, current/frozen trajectory behavior and incomplete/stale evidence rejection.

## Open work

See `activity_results.csv` for exact missing arguments and evidence routes. Source TBD/TBR values, CAD meshes and tolerances, material allowables, measured spectra, pressure/controller details, sensor/radio/battery selections, operational rules, hardware test data and reviewed specialist model outputs remain necessary. Nonlinear FEA, detailed thermochemistry/CFD, exact launch-site windows and vector-targeted Mars injection are not replaced by these reduced models.

Baseline fingerprint: `b051eeea1d4234fc323d08ad058478af361a64c4698aed13861f6f15a7454e8b`.
