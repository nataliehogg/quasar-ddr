import unittest

import numpy as np

from tools.error_models import (
    ERROR_SCENARIOS,
    ErrorScenario,
    SARM_ERROR_OPTIONS,
    UVX_EFFECTIVE_OPTIONS,
    get_error_scenario,
    resolve_distance_errors,
    resolve_sarm_error,
    resolve_uvx_error,
)


class ErrorScenarioTests(unittest.TestCase):
    def test_uvx_and_sarm_options_are_independent_and_ordered(self):
        """Catch a swapped, combined, or unordered named-tier mapping."""
        self.assertEqual(
            list(UVX_EFFECTIVE_OPTIONS),
            ["current", "future", "targeted"],
        )
        self.assertEqual(
            list(SARM_ERROR_OPTIONS),
            ["current", "future", "targeted"],
        )
        self.assertEqual(UVX_EFFECTIVE_OPTIONS["current"], 0.24)
        self.assertEqual(SARM_ERROR_OPTIONS["targeted"], 0.15)

    def test_named_uvx_uses_configured_gamma_but_custom_does_not(self):
        """Catch applying the UV--X gamma conversion to custom ln scatter."""
        named = resolve_uvx_error("current", gamma=0.55)
        expected = np.log(10.0) * 0.24 / (2.0 * (1.0 - 0.55))
        self.assertAlmostEqual(named.sigma_ln_dl, expected)
        custom = resolve_uvx_error(
            "custom", gamma=0.20, custom_sigma_ln_dl=0.31
        )
        self.assertEqual(custom.sigma_ln_dl, 0.31)
        self.assertFalse(custom.gamma_used)

    def test_sarm_resolves_without_uvx_choice(self):
        """Catch coupling a SARM option to the UV--X resolver."""
        self.assertEqual(resolve_sarm_error("future").sigma_ln_da, 0.20)
        self.assertEqual(
            resolve_sarm_error("custom", custom_sigma_ln_da=0.22).sigma_ln_da,
            0.22,
        )

    def test_named_scenarios_use_approved_inputs(self):
        expected = {
            "current": (0.24, 0.30, 0.691),
            "future": (0.17, 0.20, 0.489),
            "targeted": (0.14, 0.15, 0.403),
        }
        self.assertEqual(set(ERROR_SCENARIOS), set(expected))
        for name, (uvx, sigma_da, rounded_sigma_dl) in expected.items():
            scenario = get_error_scenario(name)
            self.assertEqual(scenario.uvx_residual_dex, uvx)
            self.assertEqual(scenario.sarm_sigma_ln, sigma_da)
            self.assertEqual(round(scenario.sigma_ln_dl, 3), rounded_sigma_dl)

    def test_custom_errors_are_natural_log_distance_scatter(self):
        resolved = resolve_distance_errors("custom", 0.31, 0.22)
        self.assertEqual(resolved.name, "custom")
        self.assertEqual(resolved.sigma_ln_dl, 0.31)
        self.assertEqual(resolved.sigma_ln_da, 0.22)
        self.assertIsNone(resolved.uvx_residual_dex)
        self.assertTrue(resolved.errors_include_intrinsic)

    def test_custom_errors_require_two_finite_non_negative_values(self):
        for dl, da in ((None, 0.2), (0.3, None), (-0.1, 0.2), (0.3, float("inf"))):
            with self.subTest(dl=dl, da=da), self.assertRaisesRegex(
                ValueError, "custom"
            ):
                resolve_distance_errors("custom", dl, da)

    def test_aggressive_is_not_a_supported_scenario(self):
        with self.assertRaisesRegex(KeyError, "current, future, targeted"):
            resolve_distance_errors("aggressive")

    def test_scenario_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "uvx_residual_dex"):
            ErrorScenario("bad", -0.1, 0.2)
        with self.assertRaisesRegex(ValueError, "sarm_sigma_ln"):
            ErrorScenario("bad", 0.2, -0.1)

    def test_scenario_rejects_nonfinite_values(self):
        for kwargs in (
            {"uvx_residual_dex": float("nan")},
            {"sarm_sigma_ln": float("inf")},
            {"gamma": float("nan")},
            {"calibration_sigma_ln": float("inf")},
        ):
            values = dict(uvx_residual_dex=0.2, sarm_sigma_ln=0.1)
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ErrorScenario("bad", **values)

    def test_unknown_scenario_lists_valid_names(self):
        with self.assertRaisesRegex(KeyError, "current"):
            get_error_scenario("unknown")


if __name__ == "__main__":
    unittest.main()
