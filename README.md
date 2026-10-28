# Hyperion

**Edit inputs → review choices → run a notebook → inspect outputs.**

Hyperion's shared 6DOF/RK45 model follows the rocket through ascent, coast,
restart, and departure. The engineering and verification notebooks use the same
saved settings and flight model.

| Open | What you can do |
|---|---|
| [01 · Simulation](01_Simulation.ipynb) | Review the vehicle and flight settings; plot the trajectory, Mach, loads, and control margins |
| [02 · Engineering checks](02_Engineering_Checks.ipynb) | Use the flight loads, compare tank thicknesses, and inspect subsystem work |
| [03 · Verification](03_Verification.ipynb) | Find failed checks, missing inputs, and evidence needed for each requirement |
| [Current results](outputs/current/summary.md) | Read the latest saved findings and follow links to plots and tables |

Each notebook follows **Inputs → Choices and assumptions → Run → Outputs →
What the results mean**. Use **Run → Run All Cells** in Jupyter. The full settings
are displayed before calculation; assumed or unresolved choices stay visible.

## Inputs: change the saved design here

| File | What it controls |
|---|---|
| [vehicle.toml](inputs/vehicle.toml) | Dimensions, masses, engines, tanks, and mass-model assumptions |
| [flight.toml](inputs/flight.toml) | Launch, atmosphere, guidance, throttle, staging, and departure |
| [checks.toml](inputs/checks.toml) | Engineering study inputs and verification acceptance rules |
| [aerodynamics.csv](inputs/aerodynamics.csv) | Mach/AoA lookup table |
| [requirements/](inputs/requirements/) | Imported requirement wording, source references, and orbital state data |
| [evidence/](inputs/evidence/README.md) | Reviewed test, inspection, demonstration, and analysis records |

TOML files are plain text with section headings and comments. Units appear in the
names or nearby notes. A shared-value reference reads the value from its owning
input file; change that original value once instead of copying it into each study.
The notebooks explain and display the saved design rather than keeping separate
physical settings in code cells. Rerun after editing an input.

Source inputs and requirements were last reconciled with OneDrive on
**28 October 2026**. The selected fairing length, CP datum, aerodynamic reference
geometry, and actuator capability still need confirmation. The notebook choices
tables show the active values and their basis.

## Outputs: see what ran

`outputs/current/` holds the readable summary, `plots/`, `tables/`, and
`inputs_used/` containing the exact saved inputs behind that run. Run metadata
identifies the calculation version. Cached flight histories are reused only while
their inputs, model, and saved files still match.

A passing calculation does **not** close a requirement. The verification notebook
separates calculation results from reviewed evidence and keeps unresolved TBD/TBR
values visible. Examples in `model/examples/` are excluded from the design checks.

## Install and run

Clone the whole repository and use Python **3.12**. The numerical packages are
pinned to the versions used for the saved results:

```sh
git clone https://github.com/m-lutter/Hyperion.git
cd Hyperion
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on macOS/Linux. Then:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

To generate the integrated simulation and verification results without Jupyter:

```sh
python -m model.verify
```

## Maintain the project

`model/` contains the shared Python calculations; `tests/` checks their behavior.
To check changes and refresh all three notebooks:

```sh
python -m unittest discover -s tests
python model/tools/execute_notebooks.py
```

After confirming OneDrive says **Backed up and synced**, import a revised
requirements workbook and rerun verification:

```sh
python model/tools/import_requirements.py "path/to/Verification Plan.xlsx"
python -m model.verify
```

The importer reads the source workbook without changing it. It does not sync
OneDrive or update vehicle inputs; reconcile those changes in `inputs/` first.

To validate the activity-to-code index without changing saved results:

```sh
python model/tools/build_verification_index.py --check
```
