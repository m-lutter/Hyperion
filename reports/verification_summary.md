# Verification results — 28 October 2026

The register covers **373 requirements and 677 activities**. It uses the inputs
synced from OneDrive on 28 October; source files and hashes are listed in
[source_manifest.json](../data/source_manifest.json).

**109 calculation functions** are assigned to **246 activities**. The remaining
activities need external analysis, test, inspection, or demonstration evidence.
No requirement is closed by a calculation alone.

## What needs attention

| Finding | Current result | Next step |
|---|---|---|
| Ascent loads | AoA is **2.177° at maximum Qα**; axial/lateral rigid-body acceleration checks pass | Validate aero inputs and coupled structural loads |
| Coast control | No RCS authority is allocated, so engine-off control fails | Define coast-control capability and rerun |
| Departure energy | The 600 s coast/restart study runs out of usable propellant at **C3 = −12.254 km²/s²** | Revisit propulsion and the departure plan; both the 7.7058 and provisional 9.8 targets are missed |
| First-stage thrust | **6,765 kN**, below the provisional **11,000 kN** limit | Reconcile the engine selection and requirement |
| Tank ullage | Source stage-1 LOX volume misses the provisional **4%** free-volume requirement | Reconcile volume, loaded mass, density, and limit |

The tank-thickness study in notebook 02 is a pressure-only design comparison.
It does not establish the installed tank's strength.

## Work remaining

| Activity result | Count | Action |
|---|---:|---|
| `EVIDENCE_REQUIRED` | 431 | Add reviewed records |
| `MISSING_INPUTS` | 227 | Supply the listed design inputs |
| `CALCULATED` | 7 | Supply approved acceptance criteria |
| `SCREEN_FAIL` | 6 | Investigate the failed checks |
| `SCREEN_PASS` | 6 | Validate assumptions and review evidence |

Requirement totals are **367 open** and **6 failed or invalid**. Open TBD/TBR
values prevent closure. See [activity_results.csv](activity_results.csv) for the
exact missing arguments and [requirement_status.csv](requirement_status.csv) for
each requirement's status.

Specialist FEA/CFD, detailed thermochemistry, hardware tests, launch-site timing,
and Mars departure-vector targeting still require their own inputs and evidence.
Reduced models do not replace those activities.

## Checks performed

- **94 tests passed**, including analytical physics cases and evidence checks.
- **Five notebooks executed without errors**, saving seven figures.
- The readability refactor preserves every published numerical screen and status
  count. It also matched 193 calculation calls and all 40 system examples against
  the previous code.

Execution details are in [notebook_execution.json](notebook_execution.json).
The exact model version is recorded in [verification_run.json](verification_run.json).
