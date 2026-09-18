import unittest

import numpy as np

from tools.ddr_likelihood import paired_distance_log_likelihood, predict_log_eta


class ModelPredictionTests(unittest.TestCase):
    def test_legacy_models_map_to_log_eta(self):
        z = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(predict_log_eta(z, "M1", alpha=-0.075), np.log(0.925))
        np.testing.assert_allclose(
            predict_log_eta(z, "M2", alpha=-0.05, beta=0.01),
            np.log(1.0 - 0.05 + 0.01 * z),
        )
        np.testing.assert_allclose(
            predict_log_eta(z, "M3", alpha=-0.134),
            -0.134 * np.log1p(z),
        )

    def test_invalid_model_is_not_silently_replaced(self):
        with self.assertRaisesRegex(ValueError, "unknown DDR model"):
            predict_log_eta([0.1], "M4", alpha=0.0)


class PairedLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.z = np.array([0.1, 0.5, 1.0])
        self.da = np.array([100.0, 500.0, 900.0])
        self.dl = 0.925 * (1.0 + self.z) ** 2 * self.da

    def test_likelihood_is_maximized_at_injected_constant(self):
        at_truth = paired_distance_log_likelihood(
            self.dl, self.da, self.z, 0.3, 0.2, "M1", alpha=-0.075
        )
        at_null = paired_distance_log_likelihood(
            self.dl, self.da, self.z, 0.3, 0.2, "M1", alpha=0.0
        )
        self.assertGreater(at_truth, at_null)

    def test_calibration_offset_can_absorb_constant_signal(self):
        with_offset = paired_distance_log_likelihood(
            self.dl,
            self.da,
            self.z,
            0.3,
            0.2,
            "M1",
            alpha=0.0,
            calibration_offset=np.log(0.925),
        )
        at_truth = paired_distance_log_likelihood(
            self.dl, self.da, self.z, 0.3, 0.2, "M1", alpha=-0.075
        )
        self.assertAlmostEqual(with_offset, at_truth)

    def test_intrinsic_scatter_is_added_once_in_log_eta_space(self):
        no_intrinsic = paired_distance_log_likelihood(
            self.dl, self.da, self.z, 0.3, 0.2, "M1", alpha=-0.075
        )
        with_intrinsic = paired_distance_log_likelihood(
            self.dl,
            self.da,
            self.z,
            0.3,
            0.2,
            "M1",
            alpha=-0.075,
            sigma_int_ln_eta=0.1,
        )
        self.assertNotEqual(no_intrinsic, with_intrinsic)


if __name__ == "__main__":
    unittest.main()
