"""Independent frame, source-table, atmosphere and mass checks.

Run from the repository root: python -m unittest discover -s tests.
"""

from pathlib import Path
import sys
import unittest
import json
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.trajectory import (
    np,
    Rotation,
    matrix,
    quaternion_derivative,
    Atmosphere,
    AeroTable,
    Vehicle,
    Simulation,
    load_config,
    run_trajectory,
)


class PhysicsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_body_rate_quaternion_convention(self):
        # Independent rotation-composition check catches body/world convention swaps.
        q = Rotation.from_euler("xyz", [0.3, -0.4, 0.7]).as_quat()
        w = np.array([0.2, -0.3, 0.4])
        dt = 1e-6
        advanced = matrix(q + quaternion_derivative(q, w) * dt)
        reference = (Rotation.from_quat(q) * Rotation.from_rotvec(w * dt)).as_matrix()
        np.testing.assert_allclose(advanced, reference, atol=2e-13, rtol=0)

    def test_initial_frame_and_earth_corotation(self):
        sim = Simulation(self.config, ROOT)
        y = sim.initial()
        q = y[6:10]
        np.testing.assert_allclose(matrix(q).T @ matrix(q), np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(matrix(q)), 1.0, places=13)
        np.testing.assert_allclose(
            matrix(q)[:, 0], y[:3] / np.linalg.norm(y[:3]), atol=1e-14
        )
        # Body angular-rate vector must map to inertial Earth's spin.
        np.testing.assert_allclose(
            matrix(q) @ y[10:13], [0, 0, 7.292115e-5], atol=1e-16
        )

    def test_standard_atmosphere_reference_points(self):
        atm = Atmosphere(
            dict(self.config["atmosphere"], temperature_offset_K=0, density_scale=1)
        )
        T, p, rho, a, vis = atm.sample(0)
        self.assertAlmostEqual(T, 288.15, places=9)
        self.assertAlmostEqual(p, 101325, places=6)
        self.assertAlmostEqual(rho, 1.225, places=5)
        # 11 km geopotential is 11019.0678 m geometric in this reference model.
        z = 6356766 * 11000 / (6356766 - 11000)
        T, p, *_ = atm.sample(z)
        self.assertAlmostEqual(T, 216.65, places=8)
        self.assertAlmostEqual(p, 22632.04, delta=0.1)
        pressures = [atm.sample(h)[1] for h in [0, 11000, 20000, 47000, 86000, 150000]]
        self.assertTrue(all(a > b > 0 for a, b in zip(pressures, pressures[1:])))

    def test_aero_force_projection_and_domain(self):
        table = AeroTable(ROOT / self.config["aero"]["file"], self.config["aero"])
        source = np.genfromtxt(
            ROOT / self.config["aero"]["file"], delimiter=",", names=True
        )
        for mach, alpha in [(0.5, 2.0), (1.0, 4.0), (2.0, 2.0), (10.0, 4.0)]:
            row = source[(source["mach"] == mach) & (source["alpha_deg"] == alpha)][0]
            ca, cn, cp, off = table.sample(mach, math.radians(alpha), True)
            a = math.radians(alpha)
            self.assertFalse(off)
            self.assertAlmostEqual(
                ca * math.cos(a) + cn * math.sin(a), row["cd_on"], places=13
            )
            self.assertAlmostEqual(
                cn * math.cos(a) - ca * math.sin(a), row["cl"], places=13
            )
            self.assertAlmostEqual(cp, row["cp_raw_in"] * 0.0254, places=12)
        self.assertFalse(table.sample(24.99, math.radians(4), True)[-1])
        self.assertTrue(table.sample(25.0, math.radians(4), True)[-1])
        self.assertTrue(table.sample(0.001, 0.0, True)[-1])
        self.assertTrue(table.sample(1.0, math.radians(4.01), True)[-1])

    def test_mass_closure_and_mass_shedding(self):
        c = self.config["vehicle"]
        v = Vehicle(c)
        p = np.array([s["propellant_kg"] for s in c["stages"]])
        mass, cg, I = v.properties(p, True, True)
        expected = (
            sum(s["dry_mass_kg"] + s["propellant_kg"] for s in c["stages"])
            + c["payload_kg"]
            + c["fairing_kg"]
        )
        self.assertAlmostEqual(mass, expected, places=7)
        self.assertTrue(0 < cg < sum(s["length_m"] for s in c["stages"]))
        self.assertTrue(np.all(I > 0))
        after = v.properties(p, False, True)
        self.assertAlmostEqual(
            mass - after[0], c["stages"][0]["dry_mass_kg"] + p[0], places=7
        )
        bare = v.properties(p, False, False)
        self.assertAlmostEqual(after[0] - bare[0], c["fairing_kg"], places=7)
        p[0] -= 1000
        self.assertAlmostEqual(mass - v.properties(p, True, True)[0], 1000, places=7)


# Frozen selected-guidance results produced before extracting the shared model.
EXPECTED = {
    "peaks": {
        "q_Pa": {
            "t_s": 63.400000000000006,
            "q_Pa": 17886.41136192289,
            "alpha_deg": 0.30578322597493507,
        },
        "qalpha_Pa_deg": {
            "t_s": 46.6,
            "qalpha_Pa_deg": 31805.859210038812,
            "alpha_deg": 2.182515784135676,
        },
        "axial_g": {
            "t_s": 217.76282742017793,
            "axial_g": 5.688415158665806,
            "alpha_deg": 2.2696286071004774,
        },
        "lateral_g": {
            "t_s": 220.96282742017786,
            "lateral_g": 0.018236264760647006,
            "alpha_deg": 3.4465482297907992,
        },
        "normal_force_N": {
            "t_s": 46.6,
            "normal_force_N": 25426.81133830988,
            "alpha_deg": 2.182515784135676,
        },
        "axial_aero_force_N": {
            "t_s": 76.4,
            "axial_aero_force_N": 161679.78339397942,
            "alpha_deg": 0.11844019351641193,
        },
        "aero_moment_Nm": {
            "t_s": 46.6,
            "aero_moment_Nm": 958519.603099112,
            "alpha_deg": 2.182515784135676,
        },
        "required_moment_Nm": {
            "t_s": 46.6,
            "required_moment_Nm": 1170525.412462395,
            "alpha_deg": 2.182515784135676,
        },
        "thrust_N": {"t_s": 0.0, "thrust_N": 6765000.0, "alpha_deg": 90.00000000000001},
        "gimbal_total_deg": {
            "t_s": 220.76282742017787,
            "gimbal_total_deg": 0.9193159460760834,
            "alpha_deg": 3.150607875931679,
        },
    },
    "cutoff_state": {
        "t_s": 472.67176984054544,
        "altitude_m": 200259.36148876045,
        "inertial_speed_m_s": 7788.334329676378,
        "vertical_speed_m_s": -0.8683205248655099,
        "osculating_perigee_m": 199526.81418327987,
        "osculating_apogee_m": 200992.07215568982,
        "prop2_kg": 14929.131874644723,
        "ideal_dv_m_s": 9707.912496032088,
        "steering_loss_m_s": 1410.6158548054402,
        "gimbal_loss_m_s": 0.030269198696382234,
        "drag_loss_m_s": 19.734389660846762,
        "gravity_loss_m_s": 897.5953015186773,
    },
    "events": [
        {"event": "liftoff", "t_s": 0.0},
        {"event": "MECO", "t_s": 217.76282742017793},
        {"event": "stage_separation", "t_s": 219.76282742017793},
        {"event": "stage2_ignition", "t_s": 219.76282742017793},
        {"event": "fairing_jettison", "t_s": 271.30919016024785},
        {"event": "orbital_speed_cutoff", "t_s": 472.67176984054544},
        {"event": "end_coast", "t_s": 502.67176984054544},
    ],
    "checks": {
        "axial_acceleration_pass": True,
        "lateral_acceleration_pass": True,
        "nominal_moment_inequality_all_samples": False,
        "nominal_moment_ratio_requirement_all_samples": False,
        "moment_demand_within_capacity_powered": True,
        "circular_200km_target_screen_pass": True,
        "validated_control_authority": False,
        "significant_q_table_domain_pass": False,
        "powered_moment_ratio_requirement_pass": True,
    },
}


BASELINE_CONFIG = {
    "description": "Hyperion current synced Oct26 inputs. 6DOF/RK45 preliminary launch-to-EPO simulation; read "
    "assumptions and MODEL_NOTES.md.",
    "input_audit": "audit/selected_inputs.json",
    "launch": {
        "latitude_deg": 28.608,
        "longitude_deg": -80.604,
        "altitude_m": 1.0,
        "azimuth_deg": 90.0,
    },
    "vehicle": {
        "diameter_m": 5.4,
        "payload_kg": 4313,
        "fairing_kg": 2155.27825,
        "fairing_length_m": 14,
        "payload_cg_above_fairing_base_m": 4.5,
        "payload_length_m": 6.0,
        "payload_radius_m": 1.5,
        "fairing_cg_above_base_m": 6.5,
        "stages": [
            {
                "name": "Stage 1",
                "dry_mass_kg": 28518.455236898277,
                "residual_kg": 12225.67377549231,
                "isp_s": 350,
                "thrust_N": 6765000,
                "min_throttle": 0.4,
                "length_m": 32.625292661945515,
                "startup_propellant_kg": 2956.4335752961897,
                "propellant_kg": 422704.5665338655,
                "engine_pivot_m": 1.5,
                "gimbal_limit_deg": 5.0,
                "gimbal_rate_deg_s": 5.0,
                "exit_area_m2": 0.0,
                "reference_pressure_Pa": 101325.0,
                "tanks": [
                    {
                        "name": "LOX",
                        "loaded_mass_kg": 330812.2694612861,
                        "bottom_m": 4.050000000000001,
                        "height_m": 12.76130694351566,
                    },
                    {
                        "name": "fuel",
                        "loaded_mass_kg": 91892.29707257946,
                        "bottom_m": 18.161306943515658,
                        "height_m": 10.413985718429851,
                    },
                ],
            },
            {
                "name": "Stage 2",
                "dry_mass_kg": 7898.353461875517,
                "residual_kg": 1917.145794837057,
                "isp_s": 451,
                "thrust_N": 890000,
                "min_throttle": 0.5,
                "length_m": 28.637720077509464,
                "startup_propellant_kg": 301.843504114127,
                "propellant_kg": 66123.84912685309,
                "engine_pivot_m": 2.0,
                "gimbal_limit_deg": 5.0,
                "gimbal_rate_deg_s": 5.0,
                "exit_area_m2": 0.0,
                "reference_pressure_Pa": 0.0,
                "tanks": [
                    {
                        "name": "LOX",
                        "loaded_mass_kg": 56677.58496587408,
                        "bottom_m": 1.8000000000000003,
                        "height_m": 3.007888642401638,
                    },
                    {
                        "name": "fuel",
                        "loaded_mass_kg": 9446.264160979013,
                        "bottom_m": 6.157888642401639,
                        "height_m": 6.679831435107824,
                    },
                ],
            },
        ],
        "payload_station_m": 51.76301273945498,
    },
    "aero": {
        "file": "inputs/aerodynamics.csv",
        "reference_area_m2": 22.902210444669596,
        "cp_station_offset_m": 0.0,
        "cp_units": "inches converted to m",
        "cp_datum": "nose; RASAero-format inference",
        "outside_domain_policy": "endpoint hold AND flag every sample",
        "post_separation_policy": "reuse full vehicle coefficients; unvalidated proxy",
    },
    "atmosphere": {
        "temperature_offset_K": 0.0,
        "density_scale": 1.0,
        "wind_reference_m_s": 7.62,
        "wind_reference_height_m": 10.0,
        "wind_exponent": 0.2,
        "wind_cap_height_m": 1000.0,
        "wind_taper_height_m": 20000.0,
        "wind_zero_height_m": 80000.0,
        "wind_toward_deg_east_of_north": 90.0,
        "gust_amplitude_m_s": 0.0,
        "gust_period_s": 12.0,
    },
    "limits": {
        "axial_g": 6.0,
        "lateral_g": 2.0,
        "control_ratio": 2.0,
        "command_axial_g": 5.8,
    },
    "control": {
        "natural_frequency_rad_s": 0.6,
        "damping_ratio": 1.0,
        "actuator_time_constant_s": 0.15,
        "rcs_moment_Nm": 0.0,
        "roll_rcs_moment_Nm": 0.0,
        "angular_acceleration_limit_deg_s2": 0.3,
        "upper_angular_acceleration_limit_deg_s2": 1.5,
    },
    "throttle": {
        "time_s": [0, 35, 65, 90, 120, 1000],
        "fraction": [1.0, 1.0, 0.8, 0.8, 1.0, 1.0],
    },
    "guidance": {
        "pitch_time_s": [0, 3, 12, 45, 90, 140, 200, 1000],
        "pitch_deg": [90, 90, 85, 85, 50, 25, 12, 0],
        "aoa_limit_min_q_Pa": 5000.0,
        "aoa_command_limit_deg": 2,
        "target_altitude_m": 200000.0,
        "altitude_gain_s2": 0.00155,
        "vertical_damping_s": 0.08,
        "post_cutoff_coast_s": 30.0,
        "aoa_limit_start_s": 25.0,
    },
    "separation": {
        "delay_s": 2.0,
        "upper_delta_v_m_s": 0.5,
        "impulse_duration_s": 0.5,
        "fairing_altitude_m": 110000.0,
    },
    "solver": {
        "method": "RK45",
        "rtol": 2e-07,
        "position_atol_m": 0.002,
        "velocity_atol_m_s": 1e-05,
        "max_step_s": 0.4,
        "output_step_s": 0.2,
        "max_time_s": 900.0,
        "verbose": True,
    },
    "assumptions": [
        "Full-vehicle aero export geometry/reference area is unverified. CP inches from nose is "
        "inferred from RASAero and tentative user confirmation.",
        "5 deg circular gimbal cone, 5 deg/s rate, 0.15 s response, all engines gimballed: design "
        "assumptions, not approved hardware capabilities.",
        "No RCS authority credited (0 Nm); powered-off control inequality may fail. Roll cannot be "
        "controlled by central-axis TVC.",
        "Equivalent tank columns, dry cylindrical masses, engine pivots, payload/fairing mass "
        "stations inferred within supplied dimensions.",
        "Fairing allocated from stage1 dry budget; adapter allocated inside upper-stage dry budget.",
        "Upper-stage and bare vehicle reuse full-stack aero table; all such samples are flagged.",
        "Deterministic legacy wind magnitude plus assumed eastward direction/taper, not a measured "
        "launch sounding.",
        "Fixed source thrust and Isp: nozzle exit area absent, pressure correction disabled "
        "explicitly.",
        "Separation impulse/delay are assumptions; no shock, flex, buffet, slosh, collision or "
        "stage-return analysis.",
        "200 km circular orbit is a provisional interpretation of the 200 km EPO target. "
        "TMI/restart and payload deployment not simulated.",
    ],
    "mission": {
        "launch_site": "Kennedy Space Center",
        "injection_strategy": "200km EPO",
        "parking_perigee_m": 200000,
        "target_C3_km2_s2": 7.7058,
        "case_A_departure_date": "2033-04-28 00:00:00",
        "time_of_flight_days": 274,
        "TMI_delta_v_m_s": 3570.5,
    },
}


class TrajectoryRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import copy

        config = copy.deepcopy(BASELINE_CONFIG)
        config["solver"]["verbose"] = False
        cls.trajectory, cls.summary, cls.simulation = run_trajectory(config)

    def test_selected_guidance_regression(self):
        # Independent frozen outputs guard against changing the model during reuse.
        for key, values in EXPECTED["peaks"].items():
            for field, expected in values.items():
                actual = self.summary["peaks"][key][field]
                self.assertAlmostEqual(
                    actual,
                    expected,
                    delta=max(1e-6, abs(expected) * 2e-8),
                    msg=f"{key}.{field}",
                )
        for field, expected in EXPECTED["cutoff_state"].items():
            self.assertAlmostEqual(
                self.summary["cutoff_state"][field],
                expected,
                delta=max(1e-5, abs(expected) * 2e-8),
                msg=field,
            )
        self.assertEqual(self.summary["checks"], EXPECTED["checks"])
        actual_events = self.simulation.events
        self.assertEqual(
            [e["event"] for e in actual_events],
            [e["event"] for e in EXPECTED["events"]],
        )
        for actual, expected in zip(actual_events, EXPECTED["events"]):
            self.assertAlmostEqual(actual["t_s"], expected["t_s"], delta=1e-5)

    def test_state_integrity_and_engineering_limitations(self):
        frame = self.trajectory
        self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).all())
        self.assertLess(abs(frame.quaternion_norm - 1).max(), 1e-7)
        self.assertLess(self.summary["peaks"]["qalpha_Pa_deg"]["alpha_deg"], 4.0)
        self.assertTrue(self.summary["checks"]["axial_acceleration_pass"])
        self.assertTrue(self.summary["checks"]["lateral_acceleration_pass"])
        # Passing the selected peak criterion must not erase known limitations.
        self.assertFalse(
            self.summary["checks"]["nominal_moment_inequality_all_samples"]
        )
        self.assertFalse(self.summary["checks"]["significant_q_table_domain_pass"])
        self.assertFalse(self.summary["checks"]["validated_control_authority"])
        self.assertTrue((frame.aero_out_of_domain == 1).any())
        self.assertTrue((frame.aero_configuration_proxy == 1).any())
        self.assertEqual(len(self.simulation.load_cases), 9)

    def test_configuration_isolation(self):
        config = load_config()
        sim = Simulation(config)
        config["guidance"]["pitch_deg"][1] = -10
        self.assertEqual(sim.c["guidance"]["pitch_deg"][1], 90)
        self.assertEqual(load_config()["guidance"]["pitch_deg"][1], 90)


if __name__ == "__main__":
    unittest.main()
