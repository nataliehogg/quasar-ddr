"""Tests for mock dataset discovery and the Cobaya configuration helper."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.qso_mock import generate_paired_log_mock, save_paired_log_mock
from tools.utilities import Utilities


def write_paired(directory, stem):
    z = np.linspace(0.2, 2.0, 8)
    da = 1000.0 * z / (1.0 + 0.5 * z)
    mock = generate_paired_log_mock(z, da, 0.925, 0.3, 0.2, mode="asimov")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    save_paired_log_mock(path / f"{stem}_paired_dataset.npz", mock)


def write_legacy_text(directory, stem):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([np.linspace(0.2, 2.0, 5), np.ones(5), np.full(5, 0.1)])
    for obs in ("dl", "da"):
        np.savetxt(path / f"{stem}_{obs}_dataset.txt", rows)


class MockDiscoveryTests(unittest.TestCase):
    def test_finds_paired_npz_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            write_paired(directory, "M1alpha-0.075beta0.0nQSO200")
            datasets, unmatched = Utilities.list_mock_datasets(directory)
            self.assertEqual(len(datasets), 1)
            self.assertEqual(unmatched, [])
            row = datasets[0]
            self.assertEqual(row["model"], "M1")
            self.assertAlmostEqual(row["alpha"], -0.075)
            self.assertEqual(row["n_QSO"], 200)
            self.assertEqual(row["settings"], "M1alpha-0.075beta0.0nQSO200")

    def test_scenario_tag_is_parsed_and_kept_in_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            write_paired(directory, "M1alpha-0.075beta0.0nQSO200_targeted")
            row = Utilities.list_mock_datasets(directory)[0][0]
            self.assertEqual(row["scenario"], "targeted")
            self.assertEqual(
                row["settings"], "M1alpha-0.075beta0.0nQSO200_targeted"
            )

    def test_untagged_datasets_report_an_unspecified_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            write_paired(directory, "M2alpha0.1beta-0.01nQSO50")
            row = Utilities.list_mock_datasets(directory)[0][0]
            self.assertEqual(row["scenario"], "unspecified")

    def test_legacy_text_pairs_are_still_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            write_legacy_text(directory, "M1alpha0.1beta0nQSO50_current")
            row = Utilities.list_mock_datasets(directory)[0][0]
            self.assertEqual(row["scenario"], "current")
            self.assertEqual(row["n_QSO"], 50)

    def test_subfolders_are_searched_and_prefix_the_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            write_paired(directory, "M1alpha-0.075beta0.0nQSO100")
            write_paired(
                Path(directory) / "alpha" / "targeted",
                "M1alpha-0.075beta0.0nQSO100",
            )
            datasets, _ = Utilities.list_mock_datasets(directory)
            settings = sorted(row["settings"] for row in datasets)
            self.assertEqual(
                settings,
                [
                    "M1alpha-0.075beta0.0nQSO100",
                    "alpha/targeted/M1alpha-0.075beta0.0nQSO100",
                ],
            )



class SamplerConfigTests(unittest.TestCase):
    def test_sampled_parameters_follow_the_model(self):
        for model, expected in (
            ("M1", ["alpha"]),
            ("M2", ["alpha", "beta"]),
            ("M3", ["alpha"]),
        ):
            dataset = {
                "model": model, "alpha": -0.075, "beta": 0.0,
                "n_QSO": 200, "settings": "stem",
            }
            info = Utilities.sampler_config(dataset, type("DDRlike", (), {}))
            self.assertEqual(sorted(info["params"]), sorted(expected))

    def test_walkers_start_at_the_injected_truth(self):
        dataset = {
            "model": "M2", "alpha": -0.075, "beta": -0.014,
            "n_QSO": 200, "settings": "stem",
        }
        info = Utilities.sampler_config(dataset, type("DDRlike", (), {}))
        self.assertEqual(info["params"]["alpha"]["ref"]["loc"], -0.075)
        self.assertEqual(info["params"]["beta"]["ref"]["loc"], -0.014)


if __name__ == "__main__":
    unittest.main()
