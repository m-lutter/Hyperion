# Hyperion

**Edit inputs → review choices → run a notebook → inspect outputs.**

Hyperion provides **19 assigned numerical analyses through five shared calculation
routes**. Its 6DOF/RK45 model follows the rocket through ascent, coast, restart,
and departure. All three notebooks use the same saved settings and flight model.

The complete verification workbook on the shared drive remains authoritative.
This repository imports the source records needed for its assigned analyses;
it does not maintain or verify the entire workbook.

| Open | What you can do |
|---|---|
| [01 · Simulation](01_Simulation.ipynb) | Review the vehicle and flight settings; plot the trajectory, Mach, loads, and control margins |
| [02 · Engineering checks](02_Engineering_Checks.ipynb) | Check tank ullage, mass allocations, installed thrust, and supplied launch-window cases |
| [03 · Verification](03_Verification.ipynb) | Review the 19 assignments, their results, coverage limits, and named follow-up work |
| [Current results](outputs/current/summary.md) | Read the latest saved findings and follow links to plots and tables |

Each notebook follows **Inputs → Choices and assumptions → Run → Outputs →
What the results mean**. Use **Run → Run All Cells** in Jupyter. The full settings
are displayed before calculation; assumed or unresolved choices stay visible.

## Inputs: change the saved design here

| File | What it controls |
|---|---|
| [vehicle.toml](inputs/vehicle.toml) | Dimensions, masses, engines, tanks, and mass-model assumptions |
| [flight.toml](inputs/flight.toml) | Launch, atmosphere, guidance, throttle, staging, and departure |
| [checks.toml](inputs/checks.toml) | Engineering analysis inputs and acceptance rules |
| [verification.toml](inputs/verification.toml) | The exact assigned activity IDs, calculation routes, artifacts, and coverage |
| [verification_gaps.csv](inputs/verification_gaps.csv) | Six bounded follow-up groups, with semester and later-development limits |
| [aerodynamics.csv](inputs/aerodynamics.csv) | Mach/AoA lookup table |
| [requirements/](inputs/requirements/) | Selected source requirements, import references, and orbital state data |
| [evidence/](inputs/evidence/README.md) | Reviewed test, inspection, demonstration, and analysis records |

TOML files are plain text with section headings and comments. Units appear in the
names or nearby notes. A shared-value reference reads the value from its owning
input file; change that original value once instead of copying it into each study.
The notebooks explain and display the saved design rather than keeping separate
physical settings in code cells. Rerun after editing an input.

The notebook choices tables show active values and their basis. The selected
fairing length, CP datum, aerodynamic reference geometry, and actuator capability
still need confirmation. The first-stage thrust threshold follows the updated
workbook value of **6.765 MN**, which remains marked TBR.

## Assigned work and coverage

`inputs/verification.toml` owns the assignment list. Each entry names one source
activity, its calculation, the expected artifact, and what the analysis covers.
Changing an acceptance rule does not add an assignment.

| Shared calculation | What it supplies |
|---|---|
| `flight_summary` | Nominal rigid-body flight, loads, control margins, and propellant use |
| `integrated_tmi` | Ascent continuation through coast, restart, and finite-burn departure |
| `lambert_window` | Departure geometry for three supplied Earth–Mars state pairs |
| `sourced_sizing` | Source mass allocations, liquid volumes, and tank ullage |
| `thrust_sum` | Installed stage thrust from the saved engine inputs |

The [follow-up list](inputs/verification_gaps.csv) keeps the semester work bounded:
mission margins, selected wind and sensitivity cases, primary/backup launch
opportunities, GNC assumptions, and critical stability/accuracy budgets.
Comprehensive Monte Carlo campaigns, operational optimization and detailed
software or flexible-body validation belong to later development.

## Outputs: see what ran

`outputs/current/` holds the readable summary, `plots/`, `tables/`, and
`inputs_used/` containing the exact saved inputs behind that run. Run metadata
identifies the calculation version. Cached flight histories are reused only while
their inputs, model, and saved files still match.

A passing calculation does **not** close a requirement. The verification notebook
reports the assigned activities and their actual failed checks, with links to
coverage and follow-up work. Parent-requirement coverage is retained separately:
other source activities and unresolved TBD/TBR values still prevent closure.

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

To generate the integrated simulation and the 19 assigned activity results without Jupyter:

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

The importer reads the source workbook without changing it and imports only the
records needed by `inputs/verification.toml`. It checks exact activity wording;
review changed assignments against the workbook before accepting them. Other
activities on those source requirements are retained only as closure context.
The importer does not sync OneDrive or update vehicle inputs; reconcile those
changes in `inputs/` first.

To validate the activity-to-code index without changing saved results:

```sh
python model/tools/build_verification_index.py --check
```
