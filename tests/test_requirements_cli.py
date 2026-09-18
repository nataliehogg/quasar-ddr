import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RequirementsCLITests(unittest.TestCase):
    def test_cli_writes_machine_readable_scenario_table(self):
        root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "requirements.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "run_requirements_forecast.py"),
                    "--eta",
                    "0.925",
                    "--output",
                    str(output),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rows = json.loads(output.read_text())
        self.assertEqual(rows[0]["scenario"], "current")
        self.assertEqual(rows[0]["n_3sigma"], 840)

    def test_cli_reports_unattainable_calibration_limited_target(self):
        root = Path(__file__).parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "run_requirements_forecast.py"),
                "--calibration-sigma-ln",
                "0.1",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rows = json.loads(completed.stdout)
        self.assertIsNone(rows[0]["n_3sigma"])


if __name__ == "__main__":
    unittest.main()
