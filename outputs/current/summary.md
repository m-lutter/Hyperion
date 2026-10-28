# Current results

These are design calculations. A passing screen does not close a requirement or qualify hardware.

[Inputs](inputs_used/inputs.csv) · [Choices](inputs_used/choices.csv) · [Source settings](inputs_used/) · [Plots](plots/) · [Result tables](tables/) · [Run details](run_manifest.json)

## Simulation

Saved 2026-10-28T22:11:56.501667+00:00

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

Saved 2026-10-28T22:12:00.339547+00:00

Saved studies:

- [tank ullage](tables/engineering_tank_ullage.csv) — 4 rows
- [mass allocation](tables/engineering_mass_allocation.csv) — 2 rows
- [installed thrust](tables/engineering_installed_thrust.csv) — 2 rows
- [load cases](tables/engineering_load_cases.csv) — 9 rows
- [control demand](tables/engineering_control_demand.csv) — 5 rows
- [mission window](tables/engineering_mission_window.csv) — 3 rows

Study results use the assumptions displayed in the notebook and saved inputs. Missing inputs and failed checks remain open in Verification.

Plots: [tank ullage](plots/engineering_tank_ullage.png)

## Verification

Saved 2026-10-28T22:12:09.933548+00:00

**19 assigned activities · 5 calculation routes.** Results saved for 19 of those activities. This is Hyperion's assigned numerical contribution; the complete shared-drive workbook remains authoritative.

[Assignment manifest](inputs_used/verification.toml) · [Activity coverage and code routes](tables/activities.csv) · [Results](tables/activity_results.csv) · [Named follow-up work](inputs_used/verification_gaps.csv)

| Activity status | Count |
| --- | --- |
| CALCULATED | 7 |
| SCREEN_PASS | 7 |
| SCREEN_FAIL | 5 |

Failed or invalid assigned activities:

| Activity | Status | Result |
| --- | --- | --- |
| MIS.0.01.A01 | SCREEN_FAIL | C3_margin_km2_s2: -22.0537 km^2/s^2; required >= 0 |
| SYS.1.02.A01 | SCREEN_FAIL | C3_margin_km2_s2: -22.0537 km^2/s^2; required >= 0 |
| SYS.1.09.A02 | SCREEN_FAIL | C3_margin_km2_s2: -22.0537 km^2/s^2; required >= 0 |
| PERF.2.13.A02 | SCREEN_FAIL | minimum_control_ratio_margin_Nm: -12766 N m; required >= 0 |
| PROP.3.TANK.02.A01 | SCREEN_FAIL | minimum_ullage_fraction: 0.0161675 fraction; required >= 0.04 |

Assigned analysis coverage:

| Artifact | Activities | Coverage |
| --- | --- | --- |
| Integrated ascent, coast and TMI simulation model | 3 | Finite-burn departure energy and propellant screen. Prograde guidance does not establish an Earth–Mars encounter or injection-vector accuracy. |
| Lambert launch-window and departure-geometry model | 1 | Three supplied state pairs. A daily launch-window sweep, launch-site access and optimization remain planned. |
| Rigid-body flight, loads and control report | 10 | Nominal rigid-body ascent with simulated state and assumed actuator limits. Flexible modes, estimator errors, hardware response and dispersed cases are outside this result. Departure attitude/control histories are reported separately in Simulation. Whole-mission control verification remains partial. |
| Source mass, liquid-volume and ullage budget | 3 | Sizing-source arithmetic. CAD geometry, component boundaries and mass allocations still require reconciliation. |
| Installed stage thrust calculation | 2 | Nominal installed-thrust arithmetic using saved engine inputs. Engine-cycle/nozzle performance and supplier acceptance remain separate. |

[Parent-requirement coverage](tables/requirement_status.csv) records assigned coverage separately from all source activities. SCREEN_PASS means the stated calculation criterion passed. Only complete, current, reviewed evidence for every source activity, with open parameters resolved, can produce REVIEWED_COMPLETE. Missing or stale evidence never passes.
