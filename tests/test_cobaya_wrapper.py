import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from tools.qso_mock import generate_paired_log_mock, save_paired_log_mock


def load_likelihood_module_without_cobaya():
    fake_cobaya = types.ModuleType("cobaya")
    fake_likelihood = types.ModuleType("cobaya.likelihood")
    fake_likelihood.Likelihood = type("Likelihood", (), {})
    sys.modules["cobaya"] = fake_cobaya
    sys.modules["cobaya.likelihood"] = fake_likelihood
    sys.modules.pop("tools.cobaya_likelihood", None)
    return importlib.import_module("tools.cobaya_likelihood")


class CobayaWrapperTests(unittest.TestCase):
    def test_moved_adapter_is_publicly_importable_without_cobaya(self):
        """Fail if the optional adapter cannot be imported at its public path."""
        module = load_likelihood_module_without_cobaya()
        self.assertTrue(issubclass(module.DDRlike, object))

    def test_wrapper_declares_every_direct_likelihood_parameter(self):
        module = load_likelihood_module_without_cobaya()
        likelihood = module.DDRlike()
        self.assertEqual(
            likelihood.get_can_support_params(),
            ("alpha", "beta", "calibration_offset", "sigma_int"),
        )

    def test_wrapper_prefers_paired_npz_and_uses_log_likelihood(self):
        module = load_likelihood_module_without_cobaya()
        z = np.array([0.1, 0.5, 1.0])
        da = np.array([100.0, 500.0, 900.0])
        mock = generate_paired_log_mock(
            z, da, eta=0.925, sigma_ln_dl=0.3,
            sigma_ln_da=0.2, mode="asimov"
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "mock")
            save_paired_log_mock(prefix + "_paired_dataset.npz", mock)
            likelihood = module.DDRlike()
            likelihood.datapath = prefix
            likelihood.model = 1
            likelihood.initialize()
            self.assertGreater(
                likelihood.logp(alpha=-0.075), likelihood.logp(alpha=0.0)
            )

    def test_wrapper_rejects_invalid_model_id(self):
        module = load_likelihood_module_without_cobaya()
        likelihood = module.DDRlike()
        likelihood.model = 4
        with self.assertRaisesRegex(ValueError, "model"):
            likelihood._model_name()

    def test_wrapper_prevents_double_counting_intrinsic_scatter(self):
        module = load_likelihood_module_without_cobaya()
        z = np.array([0.1, 0.5])
        da = np.array([100.0, 500.0])
        mock = generate_paired_log_mock(
            z, da, eta=1.0, sigma_ln_dl=0.3,
            sigma_ln_da=0.2, mode="asimov", errors_include_intrinsic=True
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "mock")
            save_paired_log_mock(prefix + "_paired_dataset.npz", mock)
            likelihood = module.DDRlike()
            likelihood.datapath = prefix
            likelihood.model = 1
            likelihood.initialize()
            with self.assertRaisesRegex(ValueError, "already include intrinsic"):
                likelihood.logp(alpha=0.0, sigma_int=0.1)


if __name__ == "__main__":
    unittest.main()
