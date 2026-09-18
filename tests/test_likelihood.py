"""Tests for the top-level DDR likelihood."""

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from tools.qso_mock import generate_paired_log_mock, save_paired_log_mock


def load_likelihood_module():
    """Import likelihood.py with a stub Cobaya base class."""
    fake_cobaya = types.ModuleType("cobaya")
    fake_likelihood = types.ModuleType("cobaya.likelihood")
    fake_likelihood.Likelihood = type("Likelihood", (), {})
    sys.modules["cobaya"] = fake_cobaya
    sys.modules["cobaya.likelihood"] = fake_likelihood
    sys.modules.pop("likelihood", None)
    return importlib.import_module("likelihood")


def build(directory, *, alpha=-0.075, sigma_ln_dl=0.3, sigma_ln_da=0.2,
          rho=0.0, mode="asimov", seed=None, n=64):
    z = np.linspace(0.2, 2.0, n)
    da = 1000.0 * z / (1.0 + 0.5 * z)
    mock = generate_paired_log_mock(
        z, da, 1.0 + alpha, sigma_ln_dl, sigma_ln_da,
        rho=rho, mode=mode, seed=seed,
    )
    prefix = str(Path(directory) / "mock")
    save_paired_log_mock(prefix + "_paired_dataset.npz", mock)
    return prefix, mock


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.module = load_likelihood_module()

    def _initialised(self, prefix, model="M1"):
        likelihood = self.module.DDRlike()
        likelihood.datapath = prefix
        likelihood.model = model
        likelihood.initialize()
        return likelihood

    def test_variance_comes_from_the_stored_log_scatters(self):
        """The variance must not depend on the realised distances.

        Deriving a log scatter as (absolute sigma / measured distance) folds each
        object's own noise into its own weight, which biases alpha by an amount
        that does not shrink with sample size.
        """
        with tempfile.TemporaryDirectory() as directory:
            prefix, mock = build(
                directory, sigma_ln_dl=0.7, sigma_ln_da=0.3,
                mode="random", seed=11,
            )
            likelihood = self._initialised(prefix)

            expected = mock.sigma_ln_dl**2 + mock.sigma_ln_da**2
            np.testing.assert_allclose(likelihood.var_log_eta, expected)

            # The value a sigma-in-Mpc column would have produced
            sigma_dl_mpc = mock.dl_true * np.sqrt(np.expm1(mock.sigma_ln_dl**2))
            sigma_da_mpc = mock.da_true * np.sqrt(np.expm1(mock.sigma_ln_da**2))
            derived = ((sigma_dl_mpc / mock.dl_observed)**2
                       + (sigma_da_mpc / mock.da_observed)**2)
            self.assertFalse(np.allclose(likelihood.var_log_eta, derived))

    def test_log_eta_matches_the_injected_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, _ = build(directory, alpha=-0.075)
            likelihood = self._initialised(prefix)
            np.testing.assert_allclose(
                likelihood.log_eta_obs, np.log(1.0 - 0.075), atol=1e-12
            )

    def test_logp_is_maximised_at_the_injected_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, _ = build(directory, alpha=-0.075)
            likelihood = self._initialised(prefix)
            best = likelihood.logp(alpha=-0.075)
            self.assertAlmostEqual(best, 0.0, places=8)
            for offset in (-0.05, 0.05, 0.15):
                self.assertLess(likelihood.logp(alpha=-0.075 + offset), best)

    def test_correlated_dataset_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, _ = build(directory, rho=0.5, mode="random", seed=3)
            with self.assertRaisesRegex(ValueError, "rho"):
                self._initialised(prefix)

    def test_unphysical_eta_scores_minus_infinity(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, _ = build(directory)
            likelihood = self._initialised(prefix)
            self.assertEqual(likelihood.logp(alpha=-1.5), -np.inf)

    def test_requirements_follow_the_model(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, _ = build(directory)
            self.assertEqual(
                set(self._initialised(prefix, "M1").get_requirements()), {"alpha"}
            )
            self.assertEqual(
                set(self._initialised(prefix, "M2").get_requirements()),
                {"alpha", "beta"},
            )

    def test_recovers_the_injected_alpha_without_bias(self):
        """Ensemble check: the estimator must centre on the injected value."""
        from scipy.optimize import minimize

        alpha, sigma_dl, sigma_da = -0.075, 0.4, 0.15
        estimates = []
        with tempfile.TemporaryDirectory() as directory:
            for seed in range(120):
                prefix, _ = build(
                    directory, alpha=alpha, sigma_ln_dl=sigma_dl,
                    sigma_ln_da=sigma_da, mode="random", seed=seed, n=300,
                )
                like = self._initialised(prefix)
                fitted = minimize(
                    lambda p: -2.0 * like.logp(alpha=p[0]),
                    [0.0], method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-12},
                ).x[0]
                estimates.append(fitted)

        estimates = np.array(estimates)
        mc_error = estimates.std() / np.sqrt(estimates.size)
        self.assertLess(abs(estimates.mean() - alpha), 4.0 * mc_error)


if __name__ == "__main__":
    unittest.main()
