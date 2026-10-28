# Current results

These are design calculations. A passing screen does not close a requirement or qualify hardware.

[Inputs](inputs_used/inputs.csv) · [Choices](inputs_used/choices.csv) · [Source settings](inputs_used/) · [Plots](plots/) · [Result tables](tables/) · [Run details](run_manifest.json)

## Simulation

Saved 2026-10-28T21:24:56.337607+00:00

| Quantity | Value | Unit | Limit | Result |
| --- | --- | --- | --- | --- |
| AoA at max-Qalpha | 2.17696 | deg | 4 | SCREEN_PASS |
| Maximum axial acceleration | 5.68841 | g | 6 | SCREEN_PASS |
| Maximum lateral acceleration | 0.0178868 | g | 2 | SCREEN_PASS |
| Maximum dynamic pressure | 17886.4 | Pa | — | CALCULATED |
| Maximum Qalpha | 31724.3 | Pa deg | — | CALCULATED |
| Steering loss | 1410.82 | m/s | — | CALCULATED |
| Minimum powered control margin | 253760 | N m | >= 0 | SCREEN_PASS |
| Minimum all-phase control margin | -12766 | N m | >= 0 | SCREEN_FAIL |

[Flight history](tables/flight_history.csv) · [Correlated load cases](tables/flight_load_cases.csv) · [Events](tables/flight_events.csv)

Control results depend on the saved actuator, aerodynamic and geometry assumptions. Coast and roll authority must be supported separately.

Departure ended with **residual_propellant_limit**. C3 = -12.254 km²/s²; target = 9.800 km²/s². This energy screen does not establish a planetary encounter.

Plots: [trajectory](plots/flight_trajectory.png) · [loads](plots/flight_loads.png) · [control and mass](plots/flight_control_and_mass.png) · [attitude and atmosphere](plots/flight_attitude_and_atmosphere.png) · [velocity and mach](plots/flight_velocity_and_mach.png) · [departure](plots/flight_departure.png)

## Engineering checks

Saved 2026-10-28T21:25:00.103845+00:00

Saved studies:

- [tank ullage](tables/engineering_tank_ullage.csv) — 4 rows
- [mass allocation](tables/engineering_mass_allocation.csv) — 2 rows
- [load cases](tables/engineering_load_cases.csv) — 9 rows
- [control demand](tables/engineering_control_demand.csv) — 5 rows
- [tank thickness](tables/engineering_tank_thickness.csv) — 6 rows
- [mission window](tables/engineering_mission_window.csv) — 3 rows
- [needed inputs](tables/engineering_needed_inputs.csv) — 6 rows

Study results use the assumptions displayed in the notebook and saved inputs. Missing inputs and failed checks remain open in Verification.

Plots: [tank thickness](plots/engineering_tank_thickness.png)

## Verification

Saved 2026-10-28T21:25:10.170065+00:00

| Activity status | Count |
| --- | --- |
| EVIDENCE_REQUIRED | 431 |
| MISSING_INPUTS | 227 |
| CALCULATED | 7 |
| SCREEN_FAIL | 6 |
| SCREEN_PASS | 6 |

| Requirement status | Count |
| --- | --- |
| OPEN | 367 |
| FAILED_OR_INVALID | 6 |

[All activity results](tables/activity_results.csv) · [Requirement status](tables/requirement_status.csv)

Failed or invalid activities:

| activity_id | status | detail |
| --- | --- | --- |
| MIS.0.01.A01 | SCREEN_FAIL | The saved coast and restart debit are screening assumptions. Shared 6DOF prograde TMI; no Mars encounter targeting. |
| SYS.1.02.A01 | SCREEN_FAIL | The saved coast and restart debit are screening assumptions. Shared 6DOF prograde TMI; no Mars encounter targeting. |
| SYS.1.09.A02 | SCREEN_FAIL | The saved coast and restart debit are screening assumptions. Shared 6DOF prograde TMI; no Mars encounter targeting. |
| PERF.2.13.A02 | SCREEN_FAIL | Requirement numeric screen only; model-validation and open source conflicts remain. |
| PROP.3.SYS.04.A02 | SCREEN_FAIL | 3 source Raptor3 engines at2255kN each; source sea-level basis remains to be substantiated. |
| PROP.3.TANK.02.A01 | SCREEN_FAIL | Current Mlox-Hlox Vehicle Sizing G4:H5,F19:G20,B31:C32; Vehicle Mass E29/E44 totals plus fairing2155.27825kg and adapter replacement845−431.976225kg. Bottom-up totals retained to source precision. |

SCREEN_PASS means the stated calculation criterion passed. Only complete, current, reviewed evidence can produce REVIEWED_COMPLETE. Missing or stale evidence never passes.
