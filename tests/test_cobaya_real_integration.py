import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from tools.qso_mock import generate_paired_log_mock, save_paired_log_mock


@unittest.skipUnless(
    importlib.util.find_spec("cobaya") is not None,
    "real Cobaya is not installed in this environment",
)
class RealCobayaIntegrationTests(unittest.TestCase):
    def test_real_model_routes_all_parameters_and_evaluates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "mock"
            mock = generate_paired_log_mock(
                np.array([0.2, 0.5, 1.0]),
                np.array([500.0, 900.0, 1200.0]),
                eta=0.925,
                sigma_ln_dl=0.3,
                sigma_ln_da=0.2,
                mode="asimov",
            )
            save_paired_log_mock(
                str(prefix) + "_paired_dataset.npz",
                mock,
            )
            script = textwrap.dedent(
                f"""
                import numpy as np
                from cobaya.model import get_model
                from tools.cobaya_likelihood import DDRlike

                info = {{
                    "params": {{
                        "alpha": {{"prior": {{"min": -0.2, "max": 0.2}}}},
                        "beta": 0.0,
                        "calibration_offset": 0.0,
                        "sigma_int": 0.0,
                    }},
                    "likelihood": {{
                        "DDRlike": {{
                            "external": DDRlike,
                            "model": 1,
                            "datapath": {str(prefix)!r},
                        }}
                    }},
                }}
                model = get_model(info)
                routed = tuple(model.likelihood["DDRlike"].input_params)
                assert routed == (
                    "alpha", "beta", "calibration_offset", "sigma_int"
                ), routed
                point = model.logposterior({{"alpha": -0.075}})
                assert np.isfinite(point.logpost), point
                print("ROUTED", routed)
                print("LOGPOST", point.logpost)
                """
            )
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = str(root / "mpl")
            environment["XDG_CONFIG_HOME"] = str(root / "config")
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ROUTED", result.stdout)
            self.assertIn("LOGPOST", result.stdout)


if __name__ == "__main__":
    unittest.main()
