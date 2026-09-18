import unittest

import numpy as np

from tools.theory import (
    Theory,
    angular_diameter_distance,
    ddr_eta,
    dimensionless_expansion_rate,
)


class CosmologyTests(unittest.TestCase):
    def test_flat_lcdm_expansion_rate(self):
        z = np.array([0.0, 1.0])
        expected = np.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)
        np.testing.assert_allclose(
            dimensionless_expansion_rate(z, "flat_lcdm", omega_m=0.3),
            expected,
        )

    def test_curved_lcdm_flat_limit_matches_flat_lcdm(self):
        z = np.array([0.1, 0.5, 1.5])
        flat = angular_diameter_distance(z, 70.0, 0.3, "flat_lcdm")
        curved_flat_limit = angular_diameter_distance(
            z, 70.0, 0.3, "lcdm", omega_k=0.0
        )
        np.testing.assert_allclose(curved_flat_limit, flat, rtol=1e-10)

    def test_flat_wcdm_w_minus_one_matches_flat_lcdm(self):
        z = np.array([0.1, 0.5, 1.5])
        lcdm = angular_diameter_distance(z, 70.0, 0.3, "flat_lcdm")
        wcdm = angular_diameter_distance(
            z, 70.0, 0.3, "flat_wcdm", w0=-1.0
        )
        np.testing.assert_allclose(wcdm, lcdm, rtol=1e-10)

    def test_distances_are_positive_finite_mpc(self):
        distances = angular_diameter_distance(
            [0.1, 1.0, 2.0], 67.36, 0.314, "flat_lcdm"
        )
        self.assertTrue(np.all(np.isfinite(distances)))
        self.assertTrue(np.all(distances > 0))
        self.assertGreater(distances[1], 1000.0)

    def test_invalid_cosmology_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "flat_lcdm"):
            angular_diameter_distance([0.5], 70.0, 0.3, "eds")


class DDRModelTests(unittest.TestCase):
    def test_m1_m2_m3_equations(self):
        z = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(ddr_eta(z, "M1", -0.075), 0.925)
        np.testing.assert_allclose(
            ddr_eta(z, "M2", -0.05, 0.01), 0.95 + 0.01 * z
        )
        np.testing.assert_allclose(
            ddr_eta(z, "M3", -0.2), (1.0 + z) ** -0.2
        )

    def test_non_positive_eta_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            ddr_eta([0.0, 2.0], "M2", alpha=-0.9, beta=-0.1)

    def test_compatibility_wrapper_uses_same_eta(self):
        theory = Theory(
            z=np.array([0.1, 1.0]),
            d_C=None,
            d_A=None,
            H=None,
            Om=None,
            H0=None,
            alpha=-0.075,
            beta=0.0,
            xi0=1.0,
            model="M1",
        )
        np.testing.assert_allclose(theory.eta, 0.925)
