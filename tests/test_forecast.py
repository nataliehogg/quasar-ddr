import math
import unittest

import numpy as np

from tools.forecast import (
    design_matrix,
    forecast_linear_ddr,
    required_sample_grid,
    scenario_detection_requirements,
)


class DesignMatrixTests(unittest.TestCase):
    def test_supported_models_have_expected_basis(self):
        z = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(design_matrix(z, "constant"), np.ones((3, 1)))
        np.testing.assert_allclose(
            design_matrix(z, "linear_z"), np.column_stack([np.ones(3), z])
        )
        np.testing.assert_allclose(
            design_matrix(z, "power_law"), np.log1p(z)[:, None]
        )

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown DDR model"):
            design_matrix([0.1], "M4")


class ForecastTests(unittest.TestCase):
    def test_asimov_forecast_recovers_injected_parameters(self):
        z = np.linspace(0.1, 2.0, 30)
        truth = np.array([-0.075, 0.02])
        result = forecast_linear_ddr(
            redshift=z,
            model="linear_z",
            injected_parameters=truth,
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
        )
        np.testing.assert_allclose(result.posterior_mean, truth, atol=1e-12)

    def test_random_error_scales_as_inverse_square_root_n(self):
        small = forecast_linear_ddr(
            np.linspace(0.1, 1.0, 20), "constant", [-0.075], 0.3, 0.2
        )
        large = forecast_linear_ddr(
            np.linspace(0.1, 1.0, 80), "constant", [-0.075], 0.3, 0.2
        )
        self.assertAlmostEqual(
            small.posterior_sigma[0] / large.posterior_sigma[0], 2.0
        )

    def test_calibration_floor_does_not_average_down(self):
        result = forecast_linear_ddr(
            np.linspace(0.1, 1.0, 100_000),
            "constant",
            [-0.075],
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            calibration_sigma=0.05,
        )
        self.assertAlmostEqual(result.posterior_sigma[0], 0.05, places=3)

    def test_nonfinite_forecast_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            forecast_linear_ddr(
                [0.1, np.nan], "constant", [-0.075], 0.3, 0.2
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            forecast_linear_ddr(
                [0.1, 0.2], "constant", [-0.075], np.inf, 0.2
            )

    def test_calibration_prior_can_be_applied_to_constant_offset(self):
        result = forecast_linear_ddr(
            np.linspace(0.1, 1.0, 20),
            "constant",
            [math.log(0.925)],
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            prior_mean=[0.0],
            prior_sigma=[0.1],
        )
        self.assertLess(result.posterior_sigma[0], 0.1)

    def test_scenario_requirement_table_reproduces_documented_values(self):
        rows = scenario_detection_requirements(eta=0.925)
        by_name = {row["scenario"]: row for row in rows}
        self.assertTrue(by_name["current"]["assumes_known_relative_calibration"])
        self.assertEqual(by_name["current"]["calibration_sigma_ln"], 0.0)
        self.assertEqual(by_name["current"]["n_3sigma"], 840)
        self.assertEqual(by_name["current"]["n_5sigma"], 2333)
        self.assertEqual(by_name["future"]["n_3sigma"], 414)
        self.assertEqual(by_name["future"]["n_5sigma"], 1150)
        self.assertEqual(by_name["targeted"]["n_3sigma"], 274)
        self.assertEqual(by_name["targeted"]["n_5sigma"], 761)
        self.assertNotIn("aggressive", by_name)

    def test_scenario_requirements_record_and_apply_uvx_gamma(self):
        """Catch accepting a UV--X gamma argument without using it in rows."""
        rows = scenario_detection_requirements(eta=0.925, gamma=0.55)
        current = {row["scenario"]: row for row in rows}["current"]
        self.assertEqual(current["uvx_gamma"], 0.55)
        self.assertAlmostEqual(
            current["sigma_ln_dl"], math.log(10.0) * 0.24 / (2.0 * 0.45)
        )

    def test_requirement_is_unattainable_above_calibration_floor(self):
        rows = scenario_detection_requirements(eta=0.925, calibration_sigma=0.1)
        self.assertFalse(rows[0]["assumes_known_relative_calibration"])
        self.assertIsNone(rows[0]["n_3sigma"])
        self.assertIsNone(rows[0]["n_5sigma"])

    def test_dense_requirement_grid_matches_scalar_benchmark(self):
        sigma_ln_dl = 0.300 * math.log(10.0)
        grid = required_sample_grid(
            eta=0.925,
            sigma_ln_dl=[sigma_ln_dl, 0.2],
            sigma_ln_da=[0.30, 0.10],
            significance=3,
        )
        self.assertEqual(grid.shape, (2, 2))
        self.assertEqual(grid[0, 0], 840)

    def test_dense_grid_marks_calibration_limited_region_infinite(self):
        grid = required_sample_grid(
            eta=0.925,
            sigma_ln_dl=[0.2],
            sigma_ln_da=[0.1],
            significance=3,
            calibration_sigma=0.1,
        )
        self.assertTrue(np.isinf(grid[0, 0]))


if __name__ == "__main__":
    unittest.main()
