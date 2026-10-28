# Hyperion

Integrated 6DOF rocket trajectory and requirements-verification tools for the
Hyperion senior design project. Requirements and design inputs were synchronized
with OneDrive on **28 October 2026**.

## Notebooks

| Notebook | Purpose |
|---|---|
| [Integrated trajectory](Hyperion_6DOF_RK45_Annotated.ipynb) | Current rigid-body RK45 ascent, inputs, position/attitude/AoA/Mach/thrust/gimbal plots and governing loads |
| [Mission and propulsion](01_Mission_and_Propulsion_Verification.ipynb) | Same 6DOF state continued through coast/restart/TMI; source Horizons Lambert cases and propulsion calculations |
| [Structures, thermal and separation](02_Structures_Thermal_and_Separation.ipynb) | Correlated trajectory loads, source pressure-sizing trade and structural/thermal model inputs |
| [Avionics, operations and program](03_Avionics_Operations_and_Program.ipynb) | Communication, electrical, control, sequencing, ground operations, cost and schedule calculations |
| [Verification register](04_Verification_Register.ipynb) | Every requirement/activity, calculation results, missing inputs and reviewed evidence status |

The notebooks are concise views of shared Python code. There is one authoritative
trajectory implementation in `hyperion/trajectory.py`; configuration and aero
data are ordinary files. No notebook contains encoded source data or duplicated
model code. Clone the **whole repository**, then run the notebooks in order.

## Run

Use Python **3.12 or newer**:

```sh
git clone https://github.com/m-lutter/Hyperion.git
cd Hyperion
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

macOS/Linux:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab
```

In the activated environment:

```sh
python -m unittest discover -s tests
python scripts/execute_notebooks.py
python -m hyperion.verify --out results/current
```

`results/` is local generated output. The flight cache is reused only when the
model, configuration, aero data **and output hashes** match. Set `RECOMPUTE=True`
in the trajectory notebook to force a new run.

## Inputs and traceability

- [`config/trajectory.json`](config/trajectory.json): vehicle, atmosphere/wind,
  guidance, limits, actuators, separation and RK45 settings.
- [`data/aero_lookup.csv`](data/aero_lookup.csv): Mach/AoA interpolation data.
- [`data/requirements.json`](data/requirements.json): all **373** requirements,
  exact activity wording, source cells, methods and open parameters.
- [`data/activities.csv`](data/activities.csv): all **677** activities and their
  calculation or external-evidence validation route.
- [`data/calculation_inputs.csv`](data/calculation_inputs.csv): required arguments
  and documentation for each reusable calculation.
- [`config/verification_inputs.json`](config/verification_inputs.json): supplied
  activity inputs, numerical acceptance criteria and their basis. Activity-ID
  entries override shared calculation inputs.
- [`data/source_manifest.json`](data/source_manifest.json): source file hashes,
  synchronization evidence and unresolved source conflicts.
- [`evidence/README.md`](evidence/README.md): test, inspection, demonstration and
  specialist-analysis record format.

Synchronized spreadsheets stay outside the repository and are never edited.
After a later revision, confirm OneDrive says **Backed up and synced**, then run:

```sh
python scripts/import_requirements.py "path/to/AE 443 Team A Requirements - Verification Plan.xlsx"
python scripts/build_verification_index.py
```

Reconcile changed vehicle inputs before rerunning. Importing requirements does not
silently change the vehicle or fetch shared-drive edits.

## Evidence scope

Every activity has an executable route: a parameterized calculation where
implemented, or validation of the actual external evidence package. **Code
coverage is not verification closure.** Specialist nonlinear FEA/CFD, detailed
engine thermochemistry, CAD collision models and hardware qualification need
their own model/test artifacts. Supplied-matrix finite element functions require
validated geometry, properties, boundary conditions and load mapping.

`SCREEN_PASS`/`SCREEN_FAIL` compare results with explicit sourced limits.
`CALCULATED` has no approved acceptance criterion. `MISSING_INPUTS` and
`EVIDENCE_REQUIRED` stay open. Only a complete set of current, reviewed passing
records can close a requirement; open TBD/TBR values prevent closure.

`config/systems_examples.json` contains illustrative fixtures, excluded from the
default evidence run. Physical test results are never inferred from simulation.

## Current findings

- Updated sizing uses an **11.5 m fairing / 58.763 m vehicle**. The master payload
  tab still says 14 m; the conflict remains open.
- Ascent reaches about **199.5 × 201.0 km**. AoA at max-Qα is **2.177°**, with
  max-Qα **31.724 kPa·deg**. Axial/lateral rigid-body screens pass.
- Powered control margin passes; all-phase control fails in coast because no RCS
  authority is allocated.
- The 600 s coast/restart/prograde finite-TMI screen reaches residual inventory
  at **C3 ≈ −12.254 km²/s²**. Both master target **7.7058** and requirement target
  **9.8** fail. Coast duration and restart debit are explicit screening assumptions;
  this energy check does not solve departure-vector targeting.
- Installed first-stage thrust is **6,765 kN**, below the provisional **11,000 kN**
  requirement. Source tank volumes also need the explicit ullage check.

CP datum, aero reference geometry, post-separation coefficients, actuator
capabilities and mass stations remain provisional. The rigid-body model does not
certify coupled loads, shock, flexibility, slosh, thermal survival, restart or no
recontact. Additional models make those analyses reproducible when the required
design data are supplied.

See [`reports/verification_summary.md`](reports/verification_summary.md) and
[`reports/notebook_execution.json`](reports/notebook_execution.json) for execution results.

## Contributing

Pull before editing and work on a branch. Update inputs/code, rerun affected tests
and notebooks, and review evidence changes before pushing. Never turn missing data
into zero or reuse an old evidence record without confirming its baseline.
