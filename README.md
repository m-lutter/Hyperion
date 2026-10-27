# Hyperion

Rocket trajectory simulation for the Hyperion senior design project.

Start with [the annotated Jupyter notebook](Hyperion_6DOF_RK45_Annotated.ipynb). It contains the editable inputs, embedded aerodynamic table, six-degree-of-freedom rigid-body model, RK45 integration, plots, and numerical checks in one file.

## Quick start

Install Python 3.12 or newer and Git. Clone this repository, then create a Python environment:

```sh
git clone https://github.com/m-lutter/Hyperion.git
cd Hyperion
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

On macOS or Linux:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab
```

Open `Hyperion_6DOF_RK45_Annotated.ipynb`. Edit the short input section, then choose **Restart Kernel and Run All**. Detailed inputs are in an expandable table; implementation and verification cells are collapsed.

The notebook reads its embedded input snapshot from its own saved file. Keep the notebook in the kernel's working folder. If you rename or move it, update `NOTEBOOK_FILE` in the first code cell. No companion data files or shared-drive connection are needed to run it. Running it does not fetch later shared-drive edits.

## Model and outputs

- Rigid-body position, quaternion attitude, and angular rates.
- Atmosphere and wind; Mach/AoA lookup interpolation with coverage flags.
- Propellant depletion, changing center of mass and inertia, stage separation, and fairing release.
- Constrained throttle and gimbal commands, payload acceleration, aerodynamic/control moments, and steering losses.
- Position, orientation, flight-path angle, AoA, Mach/speed, thrust, gimbal, and load plots, with governing peaks marked.

The included run uses the input snapshot synchronized on **26 October 2026**. Revised guidance gives **2.183° AoA at global max-Qα**, with max-Qα **31.806 kPa·deg**, 51.9% below the original guidance case. Its cutoff orbit is approximately **199.527 × 200.992 km**. The notebook was executed with all 11 code cells, six figures, physics checks, and RK45 refinement completed successfully.

## Interpretation

This is a preliminary design-screening model. AoA is below 4° at max-Qα, but early wind-incidence samples still exceed the table range. Original aerodynamic geometry/reference area and the tentative CP interpretation (inches from the nose) remain unverified. Aerodynamics after configuration changes are a flagged approximation.

Coast control is not demonstrated: engines provide no TVC torque while off, and no RCS torque is allocated. Gimbal capability and detailed mass distributions are assumptions. Structural flexibility, shock, slosh, discarded-stage clearance, restart, and TMI are outside the simulated scope. Numerical convergence does not validate these engineering assumptions.

## Working together

Pull before editing. Make changes on a branch, rerun the notebook, and review the results before committing. Preserve the embedded `hyperion_snapshot` metadata when saving; it contains the lookup data and source provenance. The optional notebook export cell writes local CSV files to `hyperion_exports/`, which is ignored by Git.

```sh
git pull --ff-only
git switch -c your-change
# Edit and rerun the notebook, then:
git add Hyperion_6DOF_RK45_Annotated.ipynb
git commit -m "Describe the trajectory change"
git push -u origin your-change
```
