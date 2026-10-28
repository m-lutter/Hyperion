"""Independent conservation and analytic-limit checks for propulsion kernels."""

import math
import unittest
import numpy as np
from hyperion.propulsion import (
    nozzle,
    integrate_consumption,
    hydraulic_network,
    ullage_transient,
    throttle_response,
    loading_time,
    liquid_ullage,
    pressurant_inventory,
    boiloff_inventory,
    tank_geometry,
    feed_transient,
    mixture_flow_transient,
    shutdown_surge,
    fill_hydraulics,
)


class PropulsionChecks(unittest.TestCase):
    def test_nozzle_pressure_thrust_and_sonic_limit(self):
        args = dict(
            chamber_pressure_Pa=1e6,
            chamber_temperature_K=3000,
            gas_constant_J_kg_K=350,
            heat_capacity_ratio=1.2,
            throat_area_m2=0.01,
            exit_area_m2=0.01,
            velocity_efficiency=1,
        )
        vacuum = nozzle(**args, ambient_pressure_Pa=0)
        sea = nozzle(**args, ambient_pressure_Pa=101325)
        self.assertAlmostEqual(vacuum["exit_mach"], 1)
        self.assertAlmostEqual(vacuum["thrust_N"] - sea["thrust_N"], 1013.25, places=7)
        self.assertAlmostEqual(vacuum["exit_temperature_K"], 3000 / 1.1)

    def test_impulse_mass_and_mixture_closure(self):
        result = integrate_consumption(
            time_s=[0, 10],
            thrust_N=[9806.65, 9806.65],
            isp_s=100,
            oxidizer_fuel_ratio=3,
        )
        self.assertAlmostEqual(result["consumed_propellant_kg"], 100)
        self.assertAlmostEqual(result["fuel_consumed_kg"], 25)
        self.assertAlmostEqual(result["oxidizer_consumed_kg"], 75)

    def test_laminar_pipe_hagen_poiseuille_limit(self):
        # Darcy with f=64/Re must reduce to the independent laminar pressure law.
        result = hydraulic_network(
            mass_flow_kg_s=0.01,
            density_kg_m3=1000,
            dynamic_viscosity_Pa_s=0.01,
            segments=[
                dict(
                    diameter_m=0.01, length_m=2, roughness_m=0, minor_loss_coefficient=0
                )
            ],
            upstream_pressure_Pa=200000,
            elevation_rise_m=0,
            gravity_m_s2=9.80665,
        )
        expected = 128 * 0.01 * 2 * (0.01 / 1000) / (math.pi * 0.01**4)
        self.assertAlmostEqual(result["friction_minor_loss_Pa"], expected, places=7)

    def test_closed_ullage_heating_energy(self):
        result = ullage_transient(
            time_s=[0, 10],
            ullage_volume_m3=[1, 1],
            inflow_kg_s=0,
            outflow_kg_s=0,
            heat_input_W=200,
            inlet_temperature_K=300,
            initial_mass_kg=2,
            initial_temperature_K=300,
            gas_constant_J_kg_K=200,
            heat_capacity_cv_J_kg_K=500,
        )
        self.assertAlmostEqual(result["temperature_K"][-1], 302, places=7)
        self.assertAlmostEqual(result["pressure_Pa"][-1], 2 * 200 * 302, places=5)

    def test_throttle_first_order_limit(self):
        result = throttle_response(
            command_time_s=[0, 5],
            command_fraction=[1, 1],
            time_constant_s=1,
            delay_s=0,
            slew_limit_per_s=100,
            initial_fraction=0,
            full_thrust_N=1000,
            sample_step_s=0.1,
        )
        self.assertAlmostEqual(
            result["actual_fraction"][-1], 1 - math.exp(-5), places=7
        )

    def test_impulse_independent_of_coarse_output_sampling(self):
        args = dict(
            command_time_s=[0, 10],
            command_fraction=[1, 1],
            time_constant_s=1,
            delay_s=0,
            slew_limit_per_s=100,
            initial_fraction=0,
            full_thrust_N=10,
        )
        coarse = throttle_response(**args, sample_step_s=10)
        fine = throttle_response(**args, sample_step_s=0.1)
        expected = 10 * (10 - 1 + math.exp(-10))
        self.assertAlmostEqual(coarse["total_impulse_N_s"], expected, places=6)
        self.assertAlmostEqual(
            coarse["total_impulse_N_s"], fine["total_impulse_N_s"], places=6
        )

    def test_inventory_checks_fail_honestly(self):
        self.assertFalse(
            liquid_ullage(
                tank_volume_m3=1, loaded_liquid_mass_kg=1100, liquid_density_kg_m3=1000
            )["geometric_fill_pass"]
        )
        with self.assertRaises(ValueError):
            loading_time(
                target_mass_kg=100,
                initial_mass_kg=0,
                delivery_mass_flow_kg_s=1,
                loss_mass_flow_kg_s=2,
            )
        result = boiloff_inventory(
            initial_liquid_mass_kg=100,
            heat_leak_W=100,
            latent_heat_J_kg=1000,
            duration_s=10,
            replenishment_kg_s=0,
        )
        self.assertAlmostEqual(result["final_liquid_mass_kg"], 99)

    def test_geometry_reduces_to_cylinder_plus_sphere(self):
        result = tank_geometry(
            inner_radius_m=2,
            cylinder_length_m=5,
            dome_height_m=2,
            dome_count=2,
            dome_shape="spherical_cap",
            loaded_liquid_mass_kg=1,
            liquid_density_kg_m3=1000,
        )
        self.assertAlmostEqual(
            result["tank_volume_m3"], math.pi * 4 * 5 + 4 * math.pi * 8 / 3
        )

    def test_feed_head_depletion_and_inertia(self):
        result = feed_transient(
            time_s=[0, 1, 2],
            mass_flow_kg_s=[1, 1, 1],
            tank_gas_pressure_Pa=200000,
            axial_acceleration_m_s2=10,
            initial_liquid_mass_kg=1000,
            tank_cross_section_m2=1,
            density_kg_m3=1000,
            dynamic_viscosity_Pa_s=0.001,
            vapor_pressure_Pa=1000,
            segments=[
                dict(
                    diameter_m=0.1, length_m=2, roughness_m=0, minor_loss_coefficient=0
                )
            ],
        )
        self.assertAlmostEqual(result["remaining_liquid_mass_kg"][-1], 998)
        self.assertAlmostEqual(
            result["feed_inlet_pressure_Pa"][0] - result["feed_inlet_pressure_Pa"][-1],
            20,
        )
        self.assertTrue(result["single_phase_pressure_screen_pass"])
        np.testing.assert_allclose(result["inertial_pressure_Pa"], 0)

    def test_equal_feed_time_constants_preserve_commanded_ratio(self):
        result = mixture_flow_transient(
            time_s=np.linspace(0, 5, 51),
            total_command_kg_s=4,
            command_oxidizer_fuel_ratio=3,
            fuel_time_constant_s=1,
            oxidizer_time_constant_s=1,
            initial_fuel_flow_kg_s=0,
            initial_oxidizer_flow_kg_s=0,
        )
        self.assertIsNone(result["actual_oxidizer_fuel_ratio"][0])
        np.testing.assert_allclose(
            result["actual_oxidizer_fuel_ratio"][1:], 3, atol=1e-12
        )
        self.assertAlmostEqual(result["fuel_flow_kg_s"][-1], 1 - math.exp(-5), places=7)

    def test_consumption_independent_of_coarse_output_sampling(self):
        args = dict(
            total_command_kg_s=10,
            command_oxidizer_fuel_ratio=1,
            fuel_time_constant_s=1,
            oxidizer_time_constant_s=1,
            initial_fuel_flow_kg_s=0,
            initial_oxidizer_flow_kg_s=0,
        )
        coarse = mixture_flow_transient(time_s=[0, 10], **args)
        fine = mixture_flow_transient(time_s=np.linspace(0, 10, 101), **args)
        expected = 5 * (10 - 1 + math.exp(-10))
        for key in ["fuel_consumed_kg", "oxidizer_consumed_kg"]:
            self.assertAlmostEqual(coarse[key], expected, places=6)
            self.assertAlmostEqual(coarse[key], fine[key], places=6)

    def test_sufficient_pressurant_mass_does_not_imply_delivery_pressure(self):
        result = pressurant_inventory(
            tank_pressure_Pa=300000,
            initial_ullage_m3=1,
            final_ullage_m3=2,
            ullage_temperature_K=300,
            gas_constant_J_kg_K=200,
            storage_volume_m3=100,
            storage_initial_pressure_Pa=100000,
            storage_final_pressure_Pa=50000,
            storage_initial_temperature_K=300,
            storage_final_temperature_K=300,
        )
        self.assertTrue(result["mass_inventory_pass"])
        self.assertFalse(result["delivery_pressure_screen_pass"])
        self.assertEqual(result["minimum_storage_to_tank_pressure_margin_Pa"], -250000)
        self.assertNotIn("inventory_pass", result)

    def test_rapid_closure_pressure_bound(self):
        result = shutdown_surge(
            density_kg_m3=1000,
            wave_speed_m_s=1000,
            initial_flow_velocity_m_s=2,
            final_flow_velocity_m_s=0,
            line_length_m=10,
            closure_time_s=0.01,
            initial_pressure_Pa=1e5,
        )
        self.assertEqual(result["pressure_change_Pa"], 2e6)
        self.assertTrue(result["rapid_closure_assumption_satisfied"])

    def test_fill_time_and_pressure_use_same_flow(self):
        result = fill_hydraulics(
            target_mass_kg=100,
            initial_mass_kg=10,
            delivery_mass_flow_kg_s=10,
            loss_mass_flow_kg_s=1,
            density_kg_m3=1000,
            dynamic_viscosity_Pa_s=0.001,
            segments=[
                dict(
                    diameter_m=0.1, length_m=1, roughness_m=0, minor_loss_coefficient=0
                )
            ],
            upstream_pressure_Pa=200000,
            elevation_rise_m=0,
            gravity_m_s2=9.80665,
        )
        self.assertEqual(result["loading"]["fill_time_s"], 10)
        self.assertAlmostEqual(
            result["hydraulics"]["segments"][0]["velocity_m_s"],
            10 / (1000 * math.pi * 0.1**2 / 4),
        )


if __name__ == "__main__":
    unittest.main()
