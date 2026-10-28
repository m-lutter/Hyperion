"""Independent orbital boundary-value, momentum, and depletion checks."""

import json
import math
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from model.mission import (
    lambert,
    relative_separation,
    finite_burn_tmi,
    rocket_equation,
    propellant_budget,
    separation_monte_carlo,
    tmi_cutoff_sensitivity,
)
from model.trajectory import Simulation, load_config, MU, RE


class MissionChecks(unittest.TestCase):
    @staticmethod
    def cutoff_case(fairing=False):
        config = load_config()
        sim = Simulation(config)
        sim.phase = 2
        sim.attached = False
        sim.fairing = fairing
        y = sim.initial()
        y[:3] = [RE + 200000, 0, 0]
        y[3:6] = [0, math.sqrt(MU / (RE + 200000)), 0]
        y[6:10] = Rotation.from_euler("z", 90, degrees=True).as_quat()
        y[10:13] = 0
        y[13] = 0
        y[14] = config["vehicle"]["stages"][1]["residual_kg"] + 0.05
        y[15:] = 0
        return config, sim.evaluate(0, y, True)

    def test_lambert_circular_quarter_and_long_way(self):
        radius = 7e6
        speed = math.sqrt(MU / radius)
        period = 2 * math.pi * math.sqrt(radius**3 / MU)
        for long_way, duration, sign in [
            (False, period / 4, 1),
            (True, period * 3 / 4, -1),
        ]:
            result = lambert(
                departure_position_m=[radius, 0, 0],
                arrival_position_m=[0, radius, 0],
                time_of_flight_s=duration,
                central_mu_m3_s2=MU,
                long_way=long_way,
            )
            np.testing.assert_allclose(
                result["departure_velocity_m_s"], [0, sign * speed, 0], atol=1e-5
            )
            self.assertLess(abs(result["time_of_flight_residual_s"]), 1e-6)
            json.dumps(result, allow_nan=False)

    def test_lambert_propagated_endpoint(self):
        r1 = np.array([5000, 10000, 2100]) * 1000.0
        r2 = np.array([-14600, 2500, 7000]) * 1000.0
        result = lambert(
            departure_position_m=r1,
            arrival_position_m=r2,
            time_of_flight_s=3600,
            central_mu_m3_s2=MU,
        )

        def rhs(t, y):
            return np.r_[y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3]

        sol = solve_ivp(
            rhs,
            (0, 3600),
            np.r_[r1, result["departure_velocity_m_s"]],
            rtol=2e-11,
            atol=1e-7,
        )
        np.testing.assert_allclose(sol.y[:3, -1], r2, atol=0.02, rtol=0)

    def test_separation_impulse_momentum(self):
        result = relative_separation(
            position_a_m=[7e6, 0, 0],
            velocity_a_m_s=[0, 0, 0],
            position_b_m=[7e6 - 2, 0, 0],
            velocity_b_m_s=[0, 0, 0],
            mass_a_kg=500,
            mass_b_kg=1000,
            impulse_on_a_N_s=[5, 0, 0],
            duration_s=1,
            required_center_clearance_m=1,
            central_mu_m3_s2=1e-10,
            sample_step_s=0.1,
        )
        np.testing.assert_allclose(
            np.array(result["delta_v_a_m_s"]) * 500
            + np.array(result["delta_v_b_m_s"]) * 1000,
            0,
            atol=1e-12,
        )
        self.assertAlmostEqual(result["center_distance_m"][-1], 2.015, places=7)
        self.assertTrue(result["sampled_clearance_pass"])

    def test_finite_tmi_stops_at_residual_and_reports_failure(self):
        config, row = self.cutoff_case()
        result = finite_burn_tmi(
            cutoff_row=row,
            config=config,
            coast_duration_s=0,
            restart_propellant_kg=0,
            target_c3_km2_s2=9.8,
            max_burn_duration_s=10,
            output_step_s=0.1,
        )
        self.assertEqual(result["status"], "residual_propellant_limit")
        self.assertFalse(result["target_C3_reached"])
        self.assertAlmostEqual(
            result["remaining_propellant_kg"], result["required_residual_kg"], places=7
        )
        self.assertAlmostEqual(result["consumed_propellant_kg"], 0.05, places=7)
        self.assertGreater(result["history"][-2]["thrust_N"], 0)
        self.assertEqual(result["history"][-1]["thrust_N"], 0)
        self.assertEqual(result["history"][-1]["phase"], 3)
        self.assertEqual(result["history"][-1]["t_s"], result["history"][-2]["t_s"])
        json.dumps(result, allow_nan=False)

    def test_retained_fairing_cannot_disappear_at_tmi(self):
        config, row = self.cutoff_case(fairing=True)
        with self.assertRaisesRegex(ValueError, "retains the fairing"):
            finite_burn_tmi(
                cutoff_row=row,
                config=config,
                coast_duration_s=0,
                restart_propellant_kg=0,
                target_c3_km2_s2=9.8,
                max_burn_duration_s=10,
            )

    def test_incompatible_cutoff_mass_is_rejected(self):
        config, row = self.cutoff_case()
        row["mass_kg"] += 100
        with self.assertRaisesRegex(ValueError, "mass/CG is incompatible"):
            finite_burn_tmi(
                cutoff_row=row,
                config=config,
                coast_duration_s=0,
                restart_propellant_kg=0,
                target_c3_km2_s2=9.8,
                max_burn_duration_s=10,
            )

    def test_already_satisfied_target_does_not_start_engine_or_debit_propellant(self):
        config, row = self.cutoff_case()
        result = finite_burn_tmi(
            cutoff_row=row,
            config=config,
            coast_duration_s=0,
            restart_propellant_kg=100,
            target_c3_km2_s2=-100,
            max_burn_duration_s=10,
        )
        self.assertEqual(result["status"], "C3_already_at_or_above_target")
        self.assertEqual(result["consumed_propellant_kg"], 0)
        self.assertEqual(result["remaining_propellant_kg"], row["prop2_kg"])
        self.assertEqual(result["elapsed_from_ascent_cutoff_s"], 0)
        self.assertTrue(all(r["thrust_N"] == 0 for r in result["history"]))
        self.assertFalse(any(e["event"] == "TMI_restart" for e in result["events"]))
        self.assertIsNone(result["powered_tracking_capacity_screen_pass"])

    def test_target_cutoff_preserves_powered_left_limit_and_engine_off_terminal(self):
        config, row = self.cutoff_case()
        initial_c3 = (
            row["inertial_speed_m_s"] ** 2 - 2 * MU / (RE + row["altitude_m"])
        ) / 1e6
        result = finite_burn_tmi(
            cutoff_row=row,
            config=config,
            coast_duration_s=0,
            restart_propellant_kg=0,
            target_c3_km2_s2=initial_c3 + 1e-6,
            max_burn_duration_s=10,
        )
        self.assertEqual(result["status"], "target_C3_reached")
        self.assertGreater(result["history"][-2]["thrust_N"], 0)
        self.assertEqual(result["history"][-1]["thrust_N"], 0)
        self.assertEqual(
            result["history"][-1]["mass_kg"], result["history"][-2]["mass_kg"]
        )

    def test_rocket_and_inventory_failure(self):
        result = rocket_equation(initial_mass_kg=100, final_mass_kg=50, isp_s=300)
        self.assertAlmostEqual(result["ideal_delta_v_m_s"], 9.80665 * 300 * math.log(2))
        result = propellant_budget(
            loaded_mass_kg=100,
            burns_kg=[90],
            startup_kg=2,
            boiloff_kg=1,
            trapped_residual_kg=5,
            required_reserve_kg=4,
        )
        self.assertFalse(result["reserve_pass"])
        self.assertEqual(result["reserve_margin_kg"], -2)

    def test_zero_dispersion_separation_reproduces_nominal(self):
        inputs = dict(
            position_a_m=[7e6, 0, 0],
            velocity_a_m_s=[0, 0, 0],
            position_b_m=[7e6 - 2, 0, 0],
            velocity_b_m_s=[0, 0, 0],
            mass_a_kg=500,
            mass_b_kg=1000,
            impulse_on_a_N_s=[5, 0, 0],
            duration_s=1,
            required_center_clearance_m=1,
            central_mu_m3_s2=1e-10,
            sample_step_s=0.1,
        )
        result = separation_monte_carlo(
            samples=3,
            seed=11,
            position_covariance_m2=np.zeros((6, 6)),
            velocity_covariance_m2=np.zeros((6, 6)),
            impulse_covariance_N2_s2=np.zeros((3, 3)),
            separation_inputs=inputs,
        )
        self.assertEqual(result["failed_samples"], 0)
        self.assertTrue(
            all(c["minimum_center_distance_m"] == 2 for c in result["cases"])
        )
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
