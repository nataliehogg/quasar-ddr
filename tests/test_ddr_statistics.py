import math
import unittest

import numpy as np

from tools.ddr_statistics import (
    equal_distance_precision_for_detection,
    gaussian_log_likelihood,
    linear_gaussian_posterior,
    log10_scatter_to_fractional_linearized,
    log_eta_covariance,
    observed_log_eta,
    required_sample_size,
    uvx_residual_dex_to_log10_distance,
)


class ErrorConversionTests(unittest.TestCase):
    def test_uvx_residual_converts_to_distance_dex(self):
        self.assertAlmostEqual(
            uvx_residual_dex_to_log10_distance(0.22, gamma=0.6),
            0.275,
        )

    def test_uvx_conversion_rejects_gamma_one(self):
        with self.assertRaisesRegex(ValueError, "gamma must be smaller than 1"):
            uvx_residual_dex_to_log10_distance(0.22, gamma=1.0)

    def test_error_conversions_reject_nonfinite_inputs(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            uvx_residual_dex_to_log10_distance(np.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            log10_scatter_to_fractional_linearized(np.inf)

    def test_log10_scatter_converts_to_linearized_fraction(self):
        self.assertAlmostEqual(
            log10_scatter_to_fractional_linearized(0.275),
            math.log(10.0) * 0.275,
        )


class LogEtaTests(unittest.TestCase):
    def test_observed_log_eta_recovers_injected_constant(self):
        z = np.array([0.1, 0.8, 2.0])
        da = np.array([100.0, 800.0, 1200.0])
        dl = 0.925 * (1.0 + z) ** 2 * da
        np.testing.assert_allclose(observed_log_eta(dl, da, z), np.log(0.925))

    def test_observed_log_eta_rejects_nonpositive_distance(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            observed_log_eta([1.0, -1.0], [1.0, 1.0], [0.1, 0.2])

    def test_observed_log_eta_rejects_nonfinite_input(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            observed_log_eta([1.0, np.nan], [1.0, 1.0], [0.1, 0.2])

    def test_covariance_includes_same_object_correlation_and_systematic_floor(self):
        covariance = log_eta_covariance(
            sigma_ln_dl=[0.3, 0.4],
            sigma_ln_da=[0.2, 0.1],
            rho=[0.5, 0.0],
            calibration_sigma=0.05,
        )
        expected_random = np.array([
            0.3**2 + 0.2**2 - 2 * 0.5 * 0.3 * 0.2,
            0.4**2 + 0.1**2,
        ])
        expected = np.diag(expected_random) + 0.05**2 * np.ones((2, 2))
        np.testing.assert_allclose(covariance, expected)


class AnalyticForecastTests(unittest.TestCase):
    def test_gaussian_log_likelihood_includes_normalization(self):
        y = np.array([0.1, -0.2])
        covariance = np.diag([0.04, 0.09])
        expected = -0.5 * (
            y @ np.linalg.solve(covariance, y)
            + np.linalg.slogdet(covariance)[1]
            + 2 * np.log(2 * np.pi)
        )
        self.assertAlmostEqual(
            gaussian_log_likelihood(y, np.zeros(2), covariance), expected
        )

    def test_gaussian_log_likelihood_rejects_indefinite_covariance(self):
        with self.assertRaisesRegex(ValueError, "positive definite"):
            gaussian_log_likelihood(
                [0.0, 0.0], [0.0, 0.0], np.diag([-1.0, -1.0])
            )

    def test_gaussian_log_likelihood_rejects_nonfinite_covariance(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            gaussian_log_likelihood(
                [0.0, 0.0], [0.0, 0.0], np.diag([1.0, np.nan])
            )

    def test_linear_gaussian_posterior_recovers_weighted_solution(self):
        design = np.column_stack([np.ones(3), np.array([0.0, 1.0, 2.0])])
        truth = np.array([-0.075, 0.02])
        y = design @ truth
        covariance = np.eye(3) * 0.04
        mean, posterior_covariance = linear_gaussian_posterior(
            design, y, covariance
        )
        np.testing.assert_allclose(mean, truth, atol=1e-12)
        np.testing.assert_allclose(
            posterior_covariance,
            np.linalg.inv(design.T @ np.linalg.solve(covariance, design)),
        )

    def test_linear_posterior_rejects_indefinite_prior(self):
        with self.assertRaisesRegex(ValueError, "prior_covariance"):
            linear_gaussian_posterior(
                np.ones((2, 1)),
                np.zeros(2),
                np.eye(2),
                prior_mean=np.zeros(1),
                prior_covariance=np.array([[-1.0]]),
            )

    def test_required_sample_size_matches_current_scenario_benchmark(self):
        signal = abs(math.log(0.925))
        sigma_ln_dl = uvx_residual_dex_to_log10_distance(0.24) * math.log(10)
        sigma_ln_eta = math.hypot(sigma_ln_dl, 0.30)
        self.assertEqual(
            required_sample_size(signal, sigma_ln_eta, significance=3),
            840,
        )

    def test_equal_precision_for_twenty_object_detection(self):
        self.assertAlmostEqual(
            equal_distance_precision_for_detection(
                eta=0.925, n_objects=20, significance=3
            ),
            0.08217868031398677,
        )


if __name__ == "__main__":
    unittest.main()
