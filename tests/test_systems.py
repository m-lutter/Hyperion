"""Analytical limits and adverse cases for reusable systems calculations."""

import json
import math
from pathlib import Path
import unittest

import numpy as np
from model import systems as s


class SystemsTests(unittest.TestCase):
    def setUp(self):
        self.examples = json.loads(
            (
                Path(__file__).resolve().parents[1] / "model/examples/systems_examples.json"
            ).read_text()
        )

    def test_all_examples_are_json_finite_and_explicit(self):
        self.assertEqual(len(s.FUNCTIONS), 40)
        self.assertEqual(set(s.FUNCTIONS), set(self.examples))
        for key, fn in s.FUNCTIONS.items():
            with self.subTest(key=key):
                self.assertTrue(self.examples[key]["example_only"])
                json.dumps(fn(**self.examples[key]["inputs"]), allow_nan=False)

    def test_free_space_inverse_square(self):
        args = self.examples["systems_rf_link"]["inputs"].copy()
        args["tx_positions_m"] = [[6471000, 0, 0], [6571000, 0, 0]]
        out = s.rf_link_budget(**args)
        self.assertAlmostEqual(
            out["received_dbw"][0] - out["received_dbw"][1], 20 * math.log10(2)
        )
        self.assertEqual(out["visible"], [True, True])
        self.assertAlmostEqual(out["elevation_deg"][0], 90)

    def test_earth_occultation_and_singular_gdop(self):
        out = s.satellite_visibility(
            [[6371000, 0, 0]], [[[26560000, 0, 0], [-26560000, 0, 0]]], -90, 6371000
        )
        self.assertEqual(out["visible"], [[True, False]])
        self.assertEqual(out["gdop"], [None])

    def test_packet_budget_includes_coding(self):
        out = s.telemetry_budget(
            [{"packet_bits": 1000, "packets_per_s": 100}], 100000, 0.5, 0.8, 0.01, 0.02
        )
        self.assertAlmostEqual(out["utilization"], 2.5)
        self.assertAlmostEqual(out["packet_latency_lower_bound_s"][0], 0.055)

    def test_energy_conversion_and_reserve(self):
        out = s.power_energy(
            [{"power_w": 100, "duration_s": 3600, "duty_fraction": 1}],
            200,
            0.9,
            0.8,
            0.1,
            0.01,
            0.02,
        )
        self.assertAlmostEqual(out["required_wh"], 125)
        self.assertAlmostEqual(out["available_wh"], 162)

    def test_dc_two_way_resistance(self):
        out = s.dc_circuit(28, 0.05, 0.01, 5, 10, 15, 0.05, 20, 0.01, 0.02)
        self.assertAlmostEqual(out["load_voltage_v"], 26.5)
        self.assertAlmostEqual(out["fault_current_a"], 140)

    def test_missing_commit_measurement_is_not_satisfied(self):
        out = s.launch_commit_rules(
            {}, [{"name": "wind", "measurement": "wind", "operator": "<=", "limit": 10}]
        )
        self.assertFalse(out["modeled_all_satisfied"])
        self.assertEqual(out["missing_measurements"], ["wind"])
        with self.assertRaises(ValueError):
            s.launch_commit_rules({}, [])

    def test_interlock_rejects_missing_guard(self):
        out = s.state_machine(
            "SAFE",
            [{"time_s": 0, "event": "fire", "signals": {}}],
            [
                {
                    "from": "SAFE",
                    "event": "fire",
                    "to": "FIRED",
                    "guards": {"armed": True},
                }
            ],
            ["FIRED"],
        )
        self.assertEqual(out["final_state"], "SAFE")
        self.assertEqual(len(out["rejected_events"]), 1)
        self.assertFalse(out["forbidden_state_visits"])

    def test_fault_tree_repeated_events_not_independent_copies(self):
        out = s.fault_tree({"A": 0.1}, {"gate": "OR", "children": ["A", "A"]})
        self.assertAlmostEqual(out["top_event_probability"], 0.1)
        out = s.fault_tree({"A": 0.1, "B": 0.2}, {"gate": "OR", "children": ["A", "B"]})
        self.assertAlmostEqual(out["top_event_probability"], 0.28)
        with self.assertRaises(ValueError):
            s.fault_tree({"A": 1.1}, "A")

    def test_critical_path_parallel_branch_float(self):
        out = s.critical_path(**self.examples["systems_critical_path"]["inputs"])
        self.assertEqual(out["duration_days"], 7)
        self.assertEqual(out["critical_tasks"], ["A", "B", "D"])
        c = next(t for t in out["tasks"] if t["id"] == "C")
        self.assertEqual(c["float_days"], 2)
        with self.assertRaises(ValueError):
            s.critical_path(
                [
                    {"id": "A", "duration_days": 1, "predecessors": ["B"]},
                    {"id": "B", "duration_days": 1, "predecessors": ["A"]},
                ]
            )

    def test_navigation_scalar_analytic_update_and_seed(self):
        args = dict(
            transition_matrix=[[1]],
            process_covariance=[[0]],
            measurement_matrix=[[1]],
            measurement_covariance=[[1]],
            initial_covariance=[[1]],
            initial_bias=[0],
            steps=1,
            samples=1000,
            seed=4,
        )
        out = s.navigation_covariance(**args)
        self.assertAlmostEqual(out["covariance"][0][0], 0.5)
        self.assertEqual(out, s.navigation_covariance(**args))
        args["process_covariance"] = [[-1]]
        with self.assertRaises(ValueError):
            s.navigation_covariance(**args)

    def test_gain_crossover_matches_analytical_second_order_loop(self):
        omega = np.geomspace(0.001, 100, 10000)
        loop = 1 / (1j * omega * (1 + 1j * omega))
        out = s.frequency_margins(omega / (2 * np.pi), loop.real, loop.imag)
        analytic_omega = math.sqrt((math.sqrt(5) - 1) / 2)
        analytic_margin = 90 - math.degrees(math.atan(analytic_omega))
        self.assertAlmostEqual(
            out["minimum_phase_margin_deg"], analytic_margin, places=4
        )
        self.assertIsNone(out["minimum_gain_margin_db"])

    def test_linear_feedback_scalar_decay(self):
        out = s.linear_feedback([[0]], [[1]], [[2]], [3], 1, 0.1)
        self.assertAlmostEqual(out["state"][-1][0], 3 * math.exp(-2))
        self.assertTrue(out["asymptotically_stable"])

    def test_lift_force_and_moment_balance(self):
        out = s.lift_reactions(1000, 4, [0, 10], [30, 30], 1.2, [10000, 10000])
        r = out["support_reaction_n"]
        w = 1000 * 9.80665 * 1.2
        self.assertAlmostEqual(sum(r), w)
        self.assertAlmostEqual(r[1] * 10, w * 4)
        with self.assertRaises(ValueError):
            s.lift_reactions(1000, 11, [0, 10], [0, 0], 1, [10000, 10000])

    def test_hold_down_force_and_moment_balance(self):
        out = s.hold_down_loads(100, 120, 4, 30, 2, 4)
        r = np.array(out["vertical_reaction_n"])
        x = 2 * np.cos(np.arange(4) * np.pi / 2)
        self.assertAlmostEqual(r.sum(), -20)
        self.assertAlmostEqual(r @ x, 30)

    def test_concentration_and_thermal_exponential(self):
        out = s.concentration_balance(100, 1, 0, 10, 0, 100, 12)
        self.assertAlmostEqual(out["final_concentration"], 10 / math.e)
        out = s.concentration_balance(100, 0, 0, 10, 1, 100, 12)
        self.assertAlmostEqual(out["final_concentration"], 11)
        out = s.thermal_response(1000, 10, 300, 280, 0, 100)
        self.assertAlmostEqual(out["final_temperature_k"], 280 + 20 / math.e)

    def test_actuator_rate_to_exponential_transition(self):
        out = s.actuator_response([0, 1, 2, 3], [5, 5, 5, 5], 0, 0.5, 2, -5, 5)
        np.testing.assert_allclose(out["position"], [0, 2, 4, 5 - math.exp(-2)])
        self.assertTrue(np.max(np.abs(out["average_rate_per_s"])) <= 2 + 1e-10)

    def test_distance_boundary_and_time_expiry(self):
        out = s.weather_separation(
            [[5, 10, 0], [5, 10, 0]], [[0, 0, 0], [10, 0, 0]], [9, 10], 10, 10
        )
        self.assertEqual(out["exclusion_active"], [True, False])

    def test_worst_case_tolerance_and_route_clearance(self):
        out = s.tolerance_stack(0.1, [0.01, -0.02], [0.005, 0.003], 0.02)
        self.assertAlmostEqual(out["worst_case_margin_m"], 0.062)
        out = s.transport_clearance([2, 3, 4], [[2.1, 4, 5]], 100, 90, 0.1)
        self.assertLess(out["minimum_clearance_margin_m"], 0)
        self.assertEqual(out["capacity_margin_kg"], -10)

    def test_pressure_exact_linear_trend(self):
        out = s.pressure_trend([0, 60, 120], [101325, 101315, 101305], 60)
        self.assertAlmostEqual(out["projected_pressure_pa"], 101295)

    def test_hydraulics_mass_time_and_surge(self):
        out = s.ground_hydraulics(
            **self.examples["systems_ground_hydraulics"]["inputs"]
        )
        self.assertAlmostEqual(out["transfer_time_s"], 100)
        self.assertAlmostEqual(out["final_storage_inventory_kg"], 4000)
        self.assertAlmostEqual(out["surge_pressure_pa"], 1000000)
        self.assertLess(out["network"]["downstream_pressure_Pa"], 500000)

    def test_sampled_yaw_changes_clearance(self):
        args = self.examples["systems_swept_handling"]["inputs"].copy()
        args["vehicle_yaw_deg"] = [0, 90]
        out = s.swept_handling(**args)
        self.assertAlmostEqual(out["sampled_rotated_envelope_m"][1][0], 35)
        self.assertAlmostEqual(out["sampled_rotated_envelope_m"][1][2], 5.4)
        self.assertLess(out["minimum_clearance_margin_m"], 0)

    def test_radiation_constant_spectrum_integral(self):
        out = s.radiation_dose(**self.examples["systems_radiation"]["inputs"])
        self.assertAlmostEqual(out["total_dose_gy"], 0.0005)
        self.assertAlmostEqual(out["dose_margin_gy"], 0.0095)

    def test_wind_force_and_base_moment(self):
        out = s.wind_strip_loads(**self.examples["systems_wind_loads"]["inputs"])
        self.assertAlmostEqual(out["base_shear_n"], 900)
        self.assertAlmostEqual(out["base_moment_nm"], 4500)
        self.assertAlmostEqual(out["bending_stress_pa"][0], 45000)

    def test_single_mast_rolling_sphere_limits(self):
        out = s.rolling_sphere(20, 45, [0, 0], [0, 20])
        self.assertAlmostEqual(out["protection_radius_m"][0], math.sqrt(1400))
        self.assertAlmostEqual(out["protection_radius_m"][1], 0)
        with self.assertRaises(ValueError):
            s.rolling_sphere(50, 45, [0], [0])

    def test_schedule_resource_balance(self):
        out = s.schedule_resources(
            **self.examples["systems_schedule_resources"]["inputs"]
        )
        self.assertAlmostEqual(out["consumption"]["nitrogen_kg"], 75)
        self.assertAlmostEqual(out["remaining_inventory"]["nitrogen_kg"], 25)

    def test_composite_consistency_rejects_mismatched_mass(self):
        args = json.loads(json.dumps(self.examples["systems_handling"]["inputs"]))
        args["lift_inputs"]["mass_kg"] = 2000
        with self.assertRaises(ValueError):
            s.handling_analysis(**args)

    def test_composite_safety_stays_false_with_missing_input(self):
        args = json.loads(
            json.dumps(self.examples["systems_safety_decision"]["inputs"])
        )
        args["rule_inputs"]["measurements"].pop("wind_m_s")
        self.assertFalse(s.safety_decision(**args)["decision"]["modeled_all_satisfied"])

    def test_forecast_bias_and_perfect_correlation(self):
        out = s.pressure_forecast_check(
            **self.examples["systems_pressure_forecast"]["inputs"]
        )
        self.assertAlmostEqual(out["forecast_bias_pa"], -10)
        self.assertAlmostEqual(out["forecast_rmse_pa"], 10)
        self.assertAlmostEqual(out["correlation"], 1)


if __name__ == "__main__":
    unittest.main()
