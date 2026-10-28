"""Independent analytical checks for structural/thermal verification kernels."""

from pathlib import Path
import sys
import math
import unittest
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyperion import structures as s


class StructuralChecks(unittest.TestCase):
    def test_von_mises_hydrostatic_and_pure_shear(self):
        self.assertEqual(float(s.von_mises([4, 4, 4, 0, 0, 0])), 0)
        self.assertAlmostEqual(
            float(s.von_mises([0, 0, 0, 10, 0, 0])), math.sqrt(3) * 10
        )

    def test_principal_section_bending(self):
        out = s.section_stress(
            axial_force_n=20,
            bending_y_nm=10,
            bending_z_nm=0,
            shear_y_n=0,
            shear_z_n=0,
            torque_nm=0,
            area_m2=2,
            iy_m4=5,
            iz_m4=5,
            torsion_j_m4=10,
            shear_area_y_m2=1,
            shear_area_z_m2=1,
            sample_yz_m=[[0, -2], [0, 2]],
            yield_allowable_pa=20,
            ultimate_allowable_pa=30,
            yield_factor=1,
            ultimate_factor=1.5,
        )
        np.testing.assert_allclose(np.array(out["stress_pa"])[:, 0], [14, 6])
        self.assertAlmostEqual(out["yield"]["utilization"], 0.7)

    def test_closed_tank_membrane(self):
        out = s.tank_stress(
            radius_m=1,
            thickness_m=0.01,
            pressure_pa=1000,
            axial_force_n=0,
            bending_moment_nm=0,
            torque_nm=0,
            yield_allowable_pa=200000,
            joint_efficiency=1,
            load_factor=1,
        )
        self.assertAlmostEqual(out["von_mises_pa"][0], math.sqrt(3) * 50000)
        self.assertTrue(out["yield"]["pass"])

    def test_euler_and_shell_knockdown(self):
        out = s.buckling(
            youngs_modulus_pa=100,
            poisson_ratio=0,
            radius_m=1,
            thickness_m=0.01,
            length_m=2,
            area_m2=0.1,
            min_second_moment_m4=0.02,
            effective_length_factor=1,
            shell_knockdown=0.5,
            column_knockdown=0.8,
            compressive_force_n=0.01,
            load_factor=2,
        )
        self.assertAlmostEqual(
            out["column_critical_load_n"], 0.8 * math.pi**2 * 100 * 0.02 / 4
        )
        self.assertAlmostEqual(out["shell_critical_stress_pa"], 0.5 / math.sqrt(3))
        eig = s.buckling_eigenvalue(
            stiffness_n_m=[[10, 0], [0, 20]],
            destabilizing_geometric_stiffness_n_m=[[2, 0], [0, 1]],
            load_factor=2,
        )
        np.testing.assert_allclose(eig["critical_load_multipliers"], [5, 20])

    def test_preloaded_joint_separation(self):
        out = s.joint_strength(
            fastener_tension_n=100,
            fastener_shear_n=30,
            fastener_tensile_area_m2=1,
            fastener_shear_area_m2=1,
            bearing_diameter_m=1,
            plate_thickness_m=1,
            tension_allowable_pa=200,
            shear_allowable_pa=100,
            bearing_allowable_pa=100,
            weld_force_n=10,
            weld_effective_throat_area_m2=1,
            weld_allowable_pa=100,
            preload_n=80,
            joint_stiffness_fraction=0.2,
            load_factor=1,
        )
        self.assertAlmostEqual(out["bolt_tension_pa"], 100)
        self.assertAlmostEqual(out["remaining_clamp_n"], 0)
        self.assertTrue(out["no_joint_separation"])

    def test_static_reactions_and_prescribed_displacement(self):
        out = s.fe_static(
            stiffness_n_m=[[10, -10], [-10, 10]],
            force_n=[0, 20],
            constrained_dofs=[0],
            prescribed_displacement_m=[1],
        )
        np.testing.assert_allclose(out["displacement_m"], [1, 3])
        np.testing.assert_allclose(out["reaction_n"], [-20, 0])
        self.assertEqual(out["free_equilibrium_residual_n"], 0)

    def test_modal_two_equal_masses(self):
        out = s.modal(stiffness_n_m=[[2, -1], [-1, 2]], mass_kg=[[1, 0], [0, 1]])
        np.testing.assert_allclose(out["eigenvalues_rad2_s2"], [1, 3], atol=1e-14)
        self.assertLess(out["mass_orthogonality_error"], 1e-14)

    def test_free_vibration_matches_cosine(self):
        t = np.linspace(0, 3, 101)
        out = s.fe_transient(
            times_s=t,
            stiffness_n_m=[[8]],
            mass_kg=[[2]],
            damping_ns_m=[[0]],
            force_n=[0],
            initial_displacement_m=[1],
            initial_velocity_m_s=[0],
        )
        np.testing.assert_allclose(
            np.array(out["displacement_m"])[:, 0], np.cos(2 * t), atol=1e-8
        )
        np.testing.assert_allclose(
            np.array(out["absolute_acceleration_m_s2"])[:, 0],
            -4 * np.cos(2 * t),
            atol=4e-8,
        )

    def test_psd_resonance_transfer(self):
        z = 0.05
        w = 2 * math.pi
        out = s.base_psd_response(
            frequency_hz=[0, 1, 2],
            base_acceleration_psd_m2_s4_hz=[1, 1, 1],
            stiffness_n_m=[[w * w]],
            mass_kg=[[1]],
            damping_ns_m=[[2 * z * w]],
            base_influence=[1],
        )
        psd = np.array(out["absolute_acceleration_psd_m2_s4_hz"])[:, 0]
        self.assertAlmostEqual(psd[0], 1)
        self.assertAlmostEqual(psd[1], (1 + 4 * z * z) / (4 * z * z))

    def test_shock_zero_and_sampling_convergence(self):
        t = np.linspace(0, 0.1, 101)
        a = np.sin(math.pi * t / 0.1)
        a[-1] = 0
        args = dict(natural_frequencies_hz=[10], damping_ratio=0.05)
        zero = s.shock_response_spectrum(times_s=t, acceleration_m_s2=t * 0, **args)
        self.assertEqual(zero["absolute_acceleration_m_s2"], [0.0])
        first = s.shock_response_spectrum(times_s=t, acceleration_m_s2=a, **args)
        t2 = np.linspace(0, 0.1, 201)
        a2 = np.sin(math.pi * t2 / 0.1)
        a2[-1] = 0
        fine = s.shock_response_spectrum(times_s=t2, acceleration_m_s2=a2, **args)
        self.assertAlmostEqual(
            first["absolute_acceleration_m_s2"][0],
            fine["absolute_acceleration_m_s2"][0],
            delta=0.001,
        )
        self.assertGreater(fine["absolute_acceleration_m_s2"][0], 1)

    def test_rainflow_repeated_triangle(self):
        out = s.fatigue_damage(
            stress_history_pa=[0, 10, 0, 10, 0],
            sn_amplitude_pa=[1, 5, 10],
            sn_cycles=[1e6, 1e3, 1e2],
            damage_limit=1,
        )
        self.assertAlmostEqual(sum(row["count"] for row in out["cycles"]), 2)
        self.assertAlmostEqual(out["miner_damage"], 0.002)
        with self.assertRaises(ValueError):
            s.fatigue_damage(
                stress_history_pa=[0, 100, 0],
                sn_amplitude_pa=[1, 5],
                sn_cycles=[1e6, 1e3],
                damage_limit=1,
            )

    def test_fracture_and_interpolation(self):
        out = s.fracture_margin(
            tensile_stress_pa=10,
            flaw_depth_m=1 / math.pi,
            geometry_factor=2,
            toughness_pa_sqrt_m=100,
            load_factor=1.5,
        )
        self.assertAlmostEqual(out["mode_i_stress_intensity_pa_sqrt_m"], 30)
        self.assertAlmostEqual(out["fracture"]["margin"], 100 / 30 - 1)
        out = s.temperature_allowable(
            temperature_k=[250],
            table_temperature_k=[200, 300],
            table_allowable_pa=[100, 80],
        )
        self.assertEqual(out["allowable_pa"], [90.0])
        with self.assertRaises(ValueError):
            s.temperature_allowable(
                temperature_k=350,
                table_temperature_k=[200, 300],
                table_allowable_pa=[100, 80],
            )

    def test_clearance_and_thermal_restraint(self):
        out = s.deformation_clearance(
            nominal_gap_m=0.1,
            displacement_toward_gap_m=[0.01, 0.02],
            tolerance_stack_m=[0.005, 0.005],
            required_clearance_m=0.07,
        )
        self.assertAlmostEqual(out["minimum_clearance_m"], 0.07)
        therm = s.thermal_strain(
            alpha_per_k=1e-5,
            temperature_change_k=100,
            length_m=2,
            youngs_modulus_pa=1e9,
            area_m2=0.01,
            restraint_fraction=1,
        )
        self.assertAlmostEqual(therm["free_expansion_m"], 0.002)
        self.assertAlmostEqual(therm["restraint_force_n"], -10000)

    def test_invalid_models_fail_closed(self):
        with self.assertRaises(ValueError):
            s.modal(stiffness_n_m=[[1, 0], [0, -1]], mass_kg=[[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            s.fe_static(stiffness_n_m=[[0]], force_n=[1], constrained_dofs=[])
        with self.assertRaises(TypeError):
            s.buckling()

    def test_spring_support_equilibrium(self):
        out = s.support_reactions(
            applied_forces_n=[[0, 0, -10]],
            application_points_m=[[0.5, 0, 0]],
            applied_moments_nm=np.zeros((0, 3)),
            support_points_m=[[-1, 0, 0], [1, 0, 0]],
            support_directions=[[0, 0, 1], [0, 0, 1]],
            support_stiffness_n_m=[100, 100],
            reference_point_m=[0, 0, 0],
        )
        np.testing.assert_allclose(
            out["support_scalar_reactions_n"], [2.5, 7.5], atol=1e-12
        )
        np.testing.assert_allclose(
            out["equilibrium_residual_force_moment"], 0, atol=1e-12
        )

    def test_rotating_discrete_mass_and_surface_pressure(self):
        out = s.inertial_loads(
            masses_kg=[2],
            positions_m=[[1, 0, 0]],
            reference_point_m=[0, 0, 0],
            reference_acceleration_m_s2=[0, 3, 0],
            angular_velocity_rad_s=[0, 0, 2],
            angular_acceleration_rad_s2=[0, 0, 1],
        )
        np.testing.assert_allclose(out["inertial_forces_n"], [[8, -8, 0]])
        np.testing.assert_allclose(out["resultant_moment_nm"], [0, 0, -8])
        surface = s.surface_loads(
            patch_pressure_pa=[10],
            patch_areas_m2=[2],
            outward_normals=[[0, 1, 0]],
            patch_centroids_m=[[1, 0, 0]],
            reference_point_m=[0, 0, 0],
        )
        np.testing.assert_allclose(surface["resultant_force_n"], [0, -20, 0])
        np.testing.assert_allclose(surface["resultant_moment_nm"], [0, 0, -20])

    def test_thermoelastic_bar_free_and_fully_restrained(self):
        recovery = np.zeros((6, 2))
        recovery[0] = [-20, 20]
        thermal = np.zeros((6, 1))
        thermal[0] = [-1]
        args = dict(
            stiffness_n_m=[[40, -40], [-40, 40]],
            mechanical_force_n=[0, 0],
            temperature_k=[310],
            stress_free_temperature_k=[300],
            thermal_force_n_k=[[-2], [2]],
            stress_recovery_pa_m=recovery,
            thermal_stress_recovery_pa_k=thermal,
        )
        free = s.thermoelastic_static(constrained_dofs=[0], **args)
        np.testing.assert_allclose(free["displacement_m"], [0, 0.5])
        np.testing.assert_allclose(free["von_mises_pa"], [0], atol=1e-14)
        fixed = s.thermoelastic_static(constrained_dofs=[0, 1], **args)
        np.testing.assert_allclose(fixed["reaction_n"], [20, -20])
        np.testing.assert_allclose(fixed["stress_pa"][0], [-10, 0, 0, 0, 0, 0])

    def test_impulse_offset_angular_momentum(self):
        out = s.impulse_momentum(
            mass_kg=2,
            inertia_kg_m2=np.diag([2, 4, 8]),
            linear_impulse_ns=[2, 0, 0],
            angular_impulse_nms=[0, 0, 0],
            impulse_offset_m=[0, 1, 0],
        )
        np.testing.assert_allclose(out["delta_velocity_m_s"], [1, 0, 0])
        np.testing.assert_allclose(out["delta_angular_velocity_rad_s"], [0, 0, -0.25])

    def test_pendulum_small_amplitude_limit(self):
        t = np.linspace(0, 3, 101)
        out = s.pendulum_slosh(
            times_s=t,
            slosh_mass_kg=2,
            pendulum_length_m=1,
            restoring_acceleration_m_s2=4,
            damping_ratio=0,
            support_acceleration_m_s2=0,
            initial_angle_rad=0.001,
            initial_rate_rad_s=0,
        )
        np.testing.assert_allclose(out["angle_rad"], 0.001 * np.cos(2 * t), atol=1e-8)
        self.assertAlmostEqual(out["small_angle_frequency_hz"], 1 / math.pi)

    def test_pressure_psd_uses_transmission_and_area(self):
        w = 2 * math.pi
        out = s.pressure_psd_response(
            frequency_hz=[0, 1, 2],
            exterior_pressure_psd_pa2_hz=[1, 1, 1],
            pressure_transmission_amplitude=[0.5, 0.5, 0.5],
            load_area_vector_m2=[2],
            stiffness_n_m=[[w * w]],
            mass_kg=[[1]],
            damping_ns_m=[[0.1 * w]],
        )
        self.assertEqual(out["acceleration_psd_m2_s4_hz"][0], [0.0])
        self.assertAlmostEqual(out["acceleration_psd_m2_s4_hz"][1][0], 100.0)
        self.assertAlmostEqual(out["interior_pressure_rms_pa"], math.sqrt(0.5))

    def test_temperature_strength_corrosion_burst_and_combinations(self):
        strength = s.temperature_strength(
            temperature_k=[250, 300],
            stress_demand_pa=[60, 60],
            table_temperature_k=[200, 300],
            table_allowable_pa=[100, 80],
            load_factor=1.5,
        )
        np.testing.assert_allclose(strength["utilization"], [1, 1.125])
        self.assertFalse(strength["pass"])
        corrosion = s.corrosion_loss(
            initial_thickness_m=0.01,
            corrosion_rate_m_s=1e-6,
            oxidation_loss_m=0.001,
            exposure_time_s=1000,
            required_remaining_thickness_m=0.009,
        )
        self.assertAlmostEqual(corrosion["remaining_thickness_m"], 0.008)
        self.assertFalse(corrosion["pass"])
        burst = s.burst_capacity(
            radius_m=1,
            thickness_m=0.01,
            material_ultimate_pa=1e6,
            weld_efficiency=0.8,
            penetration_stress_factor=2,
            required_pressure_pa=2000,
            load_factor=2,
        )
        self.assertAlmostEqual(burst["pressure_capacity_pa"], 4000)
        self.assertTrue(burst["pressure_screen"]["pass"])
        combo = s.combine_load_cases(
            case_forces_n=[[1, 0, 0], [0, 2, 0]],
            case_moments_nm=[[0, 0, 1], [0, 0, 3]],
            combination_coefficients=[[2, -1]],
        )
        self.assertEqual(combo["combined_force_n"], [[2.0, -2.0, 0.0]])
        self.assertEqual(combo["combined_moment_nm"], [[0.0, 0.0, -1.0]])

    def test_registry_and_scalar_outputs_are_jsonable(self):
        self.assertTrue(set(s.KERNEL_ACTIVITIES.values()).issubset(s.FUNCTIONS))
        out = s.strength_fracture(
            temperature_k=250,
            equivalent_stress_pa=40,
            tensile_stress_pa=20,
            table_temperature_k=[200, 300],
            table_allowable_pa=[100, 80],
            table_toughness_pa_sqrt_m=[100, 80],
            flaw_depth_m=0.01,
            geometry_factor=1,
            load_factor=1.5,
        )
        json.dumps(out, allow_nan=False)
        self.assertTrue(out["all_screens_pass"])


class ThermalChecks(unittest.TestCase):
    def test_newton_cooling_and_energy(self):
        t = np.linspace(0, 10, 101)
        out = s.thermal_network(
            times_s=t,
            heat_capacity_j_k=[10],
            conductance_w_k=[[0]],
            initial_temperature_k=[300],
            heat_input_w=[0],
            ambient_temperature_k=[280],
            ambient_conductance_w_k=[2],
            emissivity_area_m2=[0],
        )
        np.testing.assert_allclose(
            np.array(out["temperature_k"])[:, 0], 280 + 20 * np.exp(-t / 5), atol=1e-7
        )
        self.assertLess(abs(out["energy_balance_error_j"]), 1e-9)

    def test_isolated_two_node_exchange(self):
        t = np.linspace(0, 10, 51)
        out = s.thermal_network(
            times_s=t,
            heat_capacity_j_k=[10, 10],
            conductance_w_k=[[0, 2], [2, 0]],
            initial_temperature_k=[300, 400],
            heat_input_w=[0, 0],
            ambient_temperature_k=[300, 300],
            ambient_conductance_w_k=[0, 0],
            emissivity_area_m2=[0, 0],
        )
        temp = np.array(out["temperature_k"])
        np.testing.assert_allclose(temp.mean(axis=1), 350, atol=1e-10)
        np.testing.assert_allclose(
            temp[:, 1] - temp[:, 0], 100 * np.exp(-0.4 * t), atol=1e-5
        )

    def test_multilayer_equilibrium(self):
        out = s.multilayer_conduction(
            times_s=np.linspace(0, 100, 101),
            thickness_m=[0.1],
            conductivity_w_m_k=[10],
            density_kg_m3=[100],
            specific_heat_j_kg_k=[10],
            area_m2=1,
            initial_temperature_k=[350],
            left_temperature_k=400,
            right_temperature_k=300,
            left_h_w_m2_k=1000,
            right_h_w_m2_k=1000,
            heat_input_w=0,
        )
        np.testing.assert_allclose(
            np.array(out["temperature_k"])[:, 0], 350, atol=1e-10
        )

    def test_closed_vent_heating_ideal_gas_energy(self):
        t = np.linspace(0, 10, 51)
        out = s.vent_pressure(
            times_s=t,
            volume_m3=1,
            initial_pressure_pa=100000,
            initial_temperature_k=300,
            ambient_pressure_pa=100000,
            ambient_temperature_k=300,
            vent_area_m2=0,
            discharge_coefficient=0.7,
            gas_constant_j_kg_k=287,
            heat_capacity_ratio=1.4,
            heat_input_w=100,
        )
        np.testing.assert_allclose(
            out["pressure_pa"], 100000 + 0.4 * 100 * t, atol=1e-7
        )
        self.assertLess(abs(out["energy_balance_error_j"]), 1e-7)
        self.assertEqual(out["mass_balance_error_kg"], 0)

    def test_adiabatic_blowdown_relation(self):
        out = s.vent_pressure(
            times_s=np.linspace(0, 3, 101),
            volume_m3=1,
            initial_pressure_pa=200000,
            initial_temperature_k=300,
            ambient_pressure_pa=0,
            ambient_temperature_k=300,
            vent_area_m2=0.001,
            discharge_coefficient=0.7,
            gas_constant_j_kg_k=287,
            heat_capacity_ratio=1.4,
            heat_input_w=0,
        )
        mass = np.array(out["mass_kg"])
        temp = np.array(out["temperature_k"])
        np.testing.assert_allclose(temp / 300, (mass / mass[0]) ** 0.4, atol=1e-8)
        self.assertTrue(np.all(np.diff(out["pressure_pa"]) < 0))
        self.assertLess(abs(out["mass_balance_error_kg"]), 1e-12)
        self.assertLess(abs(out["energy_balance_error_j"]), 1e-7)
        json.dumps(out, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
