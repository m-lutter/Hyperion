# Hyperion

Rocket trajectory and verification calculations for the Hyperion senior design
project. The current inputs and requirements were synced with OneDrive on
**28 October 2026**.

## Start here

Open the **[Verification register](04_Verification_Register.ipynb)** for results,
failures, and missing work. Open the **[Integrated trajectory](Hyperion_6DOF_RK45_Annotated.ipynb)**
for flight plots and loads. In Jupyter, choose **Run â†’ Run All Cells**.

| Notebook | What you can do |
|---|---|
| [Integrated trajectory](Hyperion_6DOF_RK45_Annotated.ipynb) | Check inputs; plot the 6DOF/RK45 ascent, Mach, loads, and control margins |
| [Mission and propulsion](01_Mission_and_Propulsion_Verification.ipynb) | Continue through coast, restart, and departure; compare source transfer dates |
| [Structures, thermal, and separation](02_Structures_Thermal_and_Separation.ipynb) | Get flight load cases, compare tank thicknesses, and find analysis inputs |
| [Avionics, operations, and program](03_Avionics_Operations_and_Program.ipynb) | Try sequence logic and find subsystem calculations and activities |
| [Verification register](04_Verification_Register.ipynb) | Run all available checks, inspect a requirement, and export the work list |

Each notebook can run on its own. They share one trajectory model; saved flight
results are reused only while the code, inputs, and output files still match.

## Install and run

Clone the **whole repository** and use Python **3.12 or newer**:

```sh
git clone https://github.com/m-lutter/Hyperion.git
cd Hyperion
python -m venv .venv
```

Activate the environment in PowerShell with `.\.venv\Scripts\Activate.ps1`, or
on macOS/Linux with `source .venv/bin/activate`. Then:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

To run without opening Jupyter:

```sh
python -m unittest discover -s tests
python scripts/execute_notebooks.py
python -m hyperion.verify --out results/current
```

Generated files go in `results/`. Published results are in [reports/](reports/).

## Change inputs

| Change | Edit |
|---|---|
| Vehicle, wind, guidance, acceleration limits, staging, solver | [config/trajectory.json](config/trajectory.json) |
| Mach/AoA coefficients | [data/aero_lookup.csv](data/aero_lookup.csv) |
| Activity inputs, acceptance limits, and source references | [config/verification_inputs.json](config/verification_inputs.json) |
| Departure dates and planetary states | [config/mission_cases.json](config/mission_cases.json) |
| Reviewed test or analysis records | [evidence/records.json](evidence/records.json); see the [evidence guide](evidence/README.md) |

Use the units in each input name. Find a calculation's required arguments in
[calculation_inputs.csv](data/calculation_inputs.csv). In verification inputs,
an activity ID overrides shared inputs for its calculation. Notebook study
settings apply only to that notebook unless you also update the saved inputs.

## Understand the results

The register contains **373 requirements and 677 activities**. There are **109
calculation functions**, assigned to **246 activities**. Other activities need
external analysis, test, inspection, or demonstration records.

- `SCREEN_PASS` / `SCREEN_FAIL`: the calculation meets / misses a supplied limit.
- `CALCULATED`: a result exists, but no acceptance rule was supplied.
- `MISSING_INPUTS` / `EVIDENCE_REQUIRED`: more data or a reviewed record is needed.

**A passing calculation does not close a requirement.** Every activity needs
current, reviewed passing evidence; unresolved TBD/TBR values also prevent closure.
The [verification report](reports/verification_summary.md) explains the current findings.
Example inputs in `config/systems_examples.json` are excluded from verification.

The model reaches roughly **199.5 Ã— 201.0 km** with **2.177Â° AoA at maximum QÎ±**.
Axial/lateral acceleration and powered control checks pass. Departure energy,
coast control, provisional first-stage thrust, and tank ullage checks still fail.
The fairing-length conflict, CP datum, aero geometry, and actuator assumptions
remain unresolved. Flexible loads, shock, thermal survival, restart qualification,
and separation clearance need additional evidence.

## Update the requirements

Confirm OneDrive says **Backed up and synced**, then import the revised workbook:

```sh
python scripts/import_requirements.py "path/to/AE 443 Team A Requirements - Verification Plan.xlsx"
python scripts/build_verification_index.py
```

The importer only reads the workbook. It does not sync OneDrive or update vehicle
inputs. Review source conflicts in [source_manifest.json](data/source_manifest.json),
reconcile changed inputs, and rerun the calculations. Original requirement wording
and source cells are preserved in [requirements.json](data/requirements.json).

## Work on the code

The notebooks explain the studies; `hyperion/` contains the reusable functions:

- `trajectory.py`, `plotting.py`: flight equations and plots.
- `mission.py`, `propulsion.py`, `structures.py`, `systems.py`: subject calculations.
- `flight_evidence.py`, `verification.py`: flight checks and evidence status.
- `workspace.py`, `verify.py`: saved flight results and the command-line run.

Pull before editing. Keep changes on a branch, run affected tests, and rerun the
notebooks before publishing. Preserve missing values as missing; do not replace
them with zero or reuse reviewed evidence without checking its version.
