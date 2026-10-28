# Hyperion

Open **[Hyperion.ipynb](Hyperion.ipynb)** for the complete annotated simulation,
plots, engineering checks, and results for **19 assigned verification activities**.
The notebook contains the ordinary Python code, with explanations beside each
part of the model. Follow **Inputs → Choices → Model → Run → Results**.

The rigid-body **6DOF/RK45** simulation covers ascent, separation, coast, restart,
and departure. Position, attitude, speed/Mach, loads, control moments, and
propellant use appear together in the executed notebook. Engineering checks
add tank ullage, mass allocation, installed thrust, and three supplied
Earth–Mars transfer cases.

| Change or review | File |
|---|---|
| Dimensions, masses, engines, tanks | [vehicle.toml](inputs/vehicle.toml) |
| Atmosphere, wind, guidance, limits, staging, solver | [flight.toml](inputs/flight.toml) |
| Assignments, calculation inputs, acceptance rules, optional studies | [verification.toml](inputs/verification.toml) |
| Mach/AoA lookup data | [aerodynamics.csv](inputs/aerodynamics.csv) |
| Sourced requirements, orbital cases, provenance, evidence, follow-up work | [reference.json](inputs/reference.json) |

Units and assumptions are shown with the settings. Review the notebook's choices
table, edit the owning input once, then **Run → Run All Cells** and save the
notebook. Its results include failed checks: the current departure misses the
C3 target, and engine-off control authority remains unresolved. A numerical
pass does not close a parent requirement; the shared-drive verification workbook
remains authoritative.

Use **Python 3.12**. After cloning the repository, create and activate a virtual
environment, then run:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

Check model changes with `python -m unittest discover -s tests`. CI also executes
the complete notebook. To refresh requirements, first confirm OneDrive says
**Backed up and synced**, then use the notebook's explicit `import_workbook(...)`
function and rerun. The importer updates the local reference snapshot; it never
changes the source workbook or synchronizes OneDrive.
