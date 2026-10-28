"""Installed-thrust arithmetic and source liquid-volume checks."""

import unittest

from model.propulsion import liquid_ullage, thrust_sum


class PropulsionChecks(unittest.TestCase):
    def test_installed_thrust_uses_identical_engine_count(self):
        result = thrust_sum(per_engine_thrust_N=2255000, engine_count=3)
        self.assertEqual(result["installed_thrust_N"], 6765000)
        self.assertEqual(result["engine_count"], 3)
        self.assertEqual(thrust_sum(per_engine_thrust_N=0, engine_count=1)["installed_thrust_N"], 0)

    def test_engine_count_and_thrust_must_be_valid(self):
        for count in (0, -1, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                thrust_sum(per_engine_thrust_N=100, engine_count=count)
        for thrust in (-1, float("nan"), float("inf")):
            with self.subTest(thrust=thrust), self.assertRaises(ValueError):
                thrust_sum(per_engine_thrust_N=thrust, engine_count=1)

    def test_liquid_volume_and_ullage_close_the_tank_volume(self):
        result = liquid_ullage(tank_volume_m3=10, loaded_liquid_mass_kg=900, liquid_density_kg_m3=100)
        self.assertEqual(result["liquid_volume_m3"], 9)
        self.assertEqual(result["ullage_volume_m3"], 1)
        self.assertEqual(result["ullage_fraction"], 0.1)
        self.assertTrue(result["geometric_fill_pass"])

    def test_empty_full_and_overfilled_tanks_report_actual_inventory(self):
        for mass, expected_fraction, expected_pass in ((0, 1, True), (1000, 0, True), (1100, -0.1, False)):
            with self.subTest(mass=mass):
                result = liquid_ullage(tank_volume_m3=10, loaded_liquid_mass_kg=mass, liquid_density_kg_m3=100)
                self.assertAlmostEqual(result["ullage_fraction"], expected_fraction)
                self.assertEqual(result["geometric_fill_pass"], expected_pass)

    def test_liquid_inputs_reject_nonphysical_or_nonfinite_values(self):
        baseline = dict(tank_volume_m3=10, loaded_liquid_mass_kg=900, liquid_density_kg_m3=100)
        for key in baseline:
            invalid_values = (-1, float("nan"), float("inf"))
            if key != "loaded_liquid_mass_kg":
                invalid_values += (0,)
            for value in invalid_values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    liquid_ullage(**dict(baseline, **{key: value}))


if __name__ == "__main__":
    unittest.main()
