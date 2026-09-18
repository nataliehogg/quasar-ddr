import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.build_lusso2020_flux_error_model import (
    BuilderSettings,
    SourceFluxErrors,
    _broad_bin_checks,
    _promote_staged_outputs,
    build_all_outputs,
    build_flux_error_rows,
    effective_sample_size,
    make_reference_figures,
    read_lusso_table3,
    weighted_quantile,
    write_runtime_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "lusso2020_table3_small.dat"
EXPECTED_COLUMNS = (
    "z",
    "median_sigma_log_fuv_dex",
    "scatter_ln_sigma_log_fuv",
    "median_sigma_log_fx_dex",
    "scatter_ln_sigma_log_fx",
    "corr_ln_uncertainty_sizes",
    "effective_reference_count",
    "smoothing_width_z",
)


def source_from(redshift, sigma_fuv, sigma_fx):
    redshift = np.asarray(redshift, dtype=float)
    sigma_fuv = np.asarray(sigma_fuv, dtype=float)
    sigma_fx = np.asarray(sigma_fx, dtype=float)
    return SourceFluxErrors(
        redshift=redshift,
        sigma_log_fuv_dex=sigma_fuv,
        sigma_log_fx_dex=sigma_fx,
        source_sha256="0" * 64,
        total_row_count=redshift.size,
        invalid_rows=(),
    )


class FixedWidthParserTests(unittest.TestCase):
    def test_parser_uses_the_vizier_byte_ranges_and_reports_invalid_rows(self):
        source = read_lusso_table3(FIXTURE)

        self.assertEqual(source.source_column_names, ("z", "e_logFUV", "e_logFX"))
        self.assertEqual(source.valid_row_count, 12)
        self.assertEqual(source.total_row_count, 13)
        np.testing.assert_array_equal(source.redshift[[0, -1]], [0.10, 0.65])
        np.testing.assert_array_equal(
            source.sigma_log_fuv_dex[[0, -1]], [0.0100, 0.0210]
        )
        np.testing.assert_array_equal(
            source.sigma_log_fx_dex[[0, -1]], [0.0400, 0.0730]
        )
        self.assertEqual(source.invalid_rows[0].line_number, 13)
        self.assertIn("non-positive", source.invalid_rows[0].reason)
        self.assertEqual(
            source.source_sha256, hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        )

    def test_parser_reports_malformed_numeric_fields_without_realigning_columns(self):
        line = list(FIXTURE.read_text().splitlines()[0])
        line[72:79] = list(" invalid")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table3.dat"
            path.write_text("".join(line) + "\n", newline="\n")
            source = read_lusso_table3(path)

        self.assertEqual(source.valid_row_count, 0)
        self.assertEqual(source.invalid_rows[0].line_number, 1)
        self.assertIn("e_logFUV", source.invalid_rows[0].reason)


class WeightedStatisticTests(unittest.TestCase):
    def test_weighted_quantile_uses_midpoint_cumulative_weights(self):
        result = weighted_quantile(
            np.array([3.0, 1.0, 2.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([0.25, 0.50, 0.75]),
        )
        np.testing.assert_allclose(result, [1.25, 2.0, 2.75], rtol=0, atol=1e-15)
        self.assertEqual(
            weighted_quantile([1.0, 2.0, 3.0], [1.0, 8.0, 1.0], [0.5])[0],
            2.0,
        )

    def test_effective_sample_size_matches_the_weight_definition(self):
        weights = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(effective_sample_size(weights), 36.0 / 14.0)

    def test_invalid_weighted_statistic_inputs_are_rejected(self):
        invalid_cases = (
            ([1.0], [0.0]),
            ([1.0], [-1.0]),
            ([1.0, 2.0], [1.0]),
            ([1.0, np.nan], [1.0, 1.0]),
        )
        for values, weights in invalid_cases:
            with self.subTest(values=values, weights=weights):
                with self.assertRaises(ValueError):
                    weighted_quantile(values, weights, [0.5])
        with self.assertRaises(ValueError):
            effective_sample_size([0.0, 0.0])


class AdaptiveGridTests(unittest.TestCase):
    def test_grid_is_zero_anchored_and_uses_only_005_bandwidth_steps(self):
        source = source_from(
            np.linspace(0.03, 0.18, 20),
            np.linspace(0.01, 0.02, 20),
            np.linspace(0.04, 0.08, 20),
        )
        rows = build_flux_error_rows(
            source,
            BuilderSettings(
                retrieval_date="2026-07-29",
                target_effective_count=20.0,
                minimum_effective_count=1.0,
            ),
        )

        np.testing.assert_allclose([row.z for row in rows], [0.05, 0.10, 0.15])
        for row in rows:
            self.assertGreaterEqual(row.smoothing_width_z, 0.05)
            self.assertLessEqual(row.smoothing_width_z, 1.00)
            self.assertAlmostEqual(row.smoothing_width_z / 0.05, round(row.smoothing_width_z / 0.05))

    def test_dense_node_stops_at_initial_bandwidth(self):
        source = source_from(
            np.full(50, 0.50),
            np.linspace(0.01, 0.02, 50),
            np.linspace(0.04, 0.08, 50),
        )
        rows = build_flux_error_rows(
            source, BuilderSettings(retrieval_date="2026-07-29")
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].smoothing_width_z, 0.05)
        self.assertGreaterEqual(rows[0].effective_reference_count, 40.0)

    def test_sparse_node_expands_bandwidth_until_target_is_reached(self):
        source = source_from(
            np.linspace(0.30, 0.70, 50),
            np.linspace(0.01, 0.02, 50),
            np.linspace(0.04, 0.08, 50),
        )
        rows = build_flux_error_rows(
            source, BuilderSettings(retrieval_date="2026-07-29")
        )
        middle = next(row for row in rows if row.z == 0.50)

        self.assertEqual(middle.smoothing_width_z, 0.10)
        self.assertGreaterEqual(middle.effective_reference_count, 40.0)

    def test_unreachable_target_uses_maximum_bandwidth_and_trims_unsupported_edge(self):
        source = source_from(
            np.r_[np.full(12, 0.50), 8.0],
            np.linspace(0.01, 0.03, 13),
            np.linspace(0.04, 0.09, 13),
        )
        rows = build_flux_error_rows(
            source, BuilderSettings(retrieval_date="2026-07-29")
        )

        self.assertTrue(rows)
        self.assertLess(rows[-1].z, 8.0)
        self.assertTrue(all(row.effective_reference_count >= 10.0 for row in rows))
        self.assertTrue(
            any(
                row.smoothing_width_z == 1.0
                and row.effective_reference_count < 40.0
                for row in rows
            )
        )

    def test_rows_use_robust_log_spread_and_log_space_correlation(self):
        log_fuv = np.array([0.0, 1.0, 2.0])
        log_fx = 4.0 - 2.0 * log_fuv
        source = source_from(
            np.full(3, 0.5), np.exp(log_fuv), np.exp(log_fx)
        )
        rows = build_flux_error_rows(
            source,
            BuilderSettings(
                retrieval_date="2026-07-29",
                target_effective_count=3.0,
                minimum_effective_count=1.0,
            ),
        )

        self.assertAlmostEqual(rows[0].median_sigma_log_fuv_dex, math.e)
        self.assertAlmostEqual(rows[0].scatter_ln_sigma_log_fuv, 1.0)
        self.assertAlmostEqual(rows[0].scatter_ln_sigma_log_fx, 2.0)
        self.assertAlmostEqual(rows[0].corr_ln_uncertainty_sizes, -1.0)

    def test_correlation_winsorizes_log_uncertainty_outliers(self):
        log_fuv = np.linspace(0.0, 1.0, 101)
        log_fx = log_fuv.copy()
        log_fx[-1] = 8.0
        source = source_from(
            np.full(101, 0.5), np.exp(log_fuv), np.exp(log_fx)
        )
        rows = build_flux_error_rows(
            source, BuilderSettings(retrieval_date="2026-07-29")
        )

        self.assertAlmostEqual(
            rows[0].corr_ln_uncertainty_sizes, 0.7218263338430985
        )


class ArtifactAndCliTests(unittest.TestCase):
    def setUp(self):
        self.source = read_lusso_table3(FIXTURE)
        self.settings = BuilderSettings(retrieval_date="2026-07-29")
        self.rows = build_flux_error_rows(self.source, self.settings)

    def test_csv_schema_format_and_provenance_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_result = write_runtime_artifacts(
                self.rows, self.source, self.settings, Path(first)
            )
            second_result = write_runtime_artifacts(
                self.rows, self.source, self.settings, Path(second)
            )
            first_csv = first_result.csv_path.read_bytes()
            first_metadata = first_result.metadata_path.read_bytes()

            self.assertEqual(first_csv, second_result.csv_path.read_bytes())
            self.assertEqual(first_metadata, second_result.metadata_path.read_bytes())
            self.assertNotIn(b"\r\n", first_csv)
            lines = first_csv.decode("ascii").splitlines()
            self.assertEqual(tuple(lines[0].split(",")), EXPECTED_COLUMNS)
            for field in lines[1].split(","):
                integer, fractional = field.split(".")
                self.assertTrue(integer.lstrip("-").isdigit())
                self.assertEqual(len(fractional), 10)
            metadata = json.loads(first_metadata)
            self.assertEqual(metadata["table_schema_version"], 1)
            self.assertEqual(metadata["source_sha256"], self.source.source_sha256)
            self.assertEqual(
                metadata["csv_sha256"], hashlib.sha256(first_csv).hexdigest()
            )
            self.assertNotIn("output_path", metadata)
            self.assertEqual(metadata["retrieval_date"], "2026-07-29")

    def test_writer_refuses_to_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_runtime_artifacts(self.rows, self.source, self.settings, output)
            with self.assertRaises(FileExistsError):
                write_runtime_artifacts(self.rows, self.source, self.settings, output)

    def test_reference_figure_builder_creates_both_pdfs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_paths = make_reference_figures(
                self.source, self.rows, Path(first)
            )
            second_paths = make_reference_figures(
                self.source, self.rows, Path(second)
            )
            self.assertEqual(
                {path.name for path in first_paths},
                {
                    "lusso2020_flux_error_reference.pdf",
                    "lusso2020_propagated_distance_uncertainty.pdf",
                },
            )
            for path, second_path in zip(first_paths, second_paths):
                self.assertTrue(path.read_bytes().startswith(b"%PDF"))
                self.assertEqual(path.read_bytes(), second_path.read_bytes())
            with self.assertRaises(FileExistsError):
                make_reference_figures(self.source, self.rows, Path(first))

    def test_reference_figure_has_table_based_propagated_catalogue_panel(self):
        """Catch removal of the real-source third panel from Figure 1."""
        with tempfile.TemporaryDirectory() as temporary:
            reference_path, _ = make_reference_figures(
                self.source, self.rows, Path(temporary)
            )
            extracted_text = subprocess.check_output(
                ["gs", "-dBATCH", "-dNOPAUSE", "-sDEVICE=txtwrite", "-sOutputFile=-", str(reference_path)],
                text=True,
            )

        self.assertIn("Table-based propagated uncertainty", extracted_text)
        self.assertIn("Lusso et al. (2020) catalogue", extracted_text)
        self.assertIn("table-based conditional median", extracted_text)
        self.assertIn("Current", extracted_text)
        self.assertIn("Future", extracted_text)
        self.assertIn("Targeted", extracted_text)
        self.assertNotIn("simulated", extracted_text.lower())

    def test_late_figure_failure_preserves_every_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            figures = Path(temporary) / "figures"
            output.mkdir()
            figures.mkdir()
            expected = {}
            for path in (
                output / "lusso2020_flux_error_model.csv",
                output / "lusso2020_flux_error_model_metadata.json",
                figures / "lusso2020_flux_error_reference.pdf",
                figures / "lusso2020_propagated_distance_uncertainty.pdf",
            ):
                expected[path] = f"existing:{path.name}".encode("ascii")
                path.write_bytes(expected[path])

            with mock.patch(
                "scripts.build_lusso2020_flux_error_model.make_reference_figures",
                side_effect=RuntimeError("forced late plotting failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced late"):
                    build_all_outputs(
                        FIXTURE,
                        BuilderSettings(
                            retrieval_date="2026-07-29", overwrite=True
                        ),
                        output,
                        figures,
                    )

            self.assertEqual(
                {path: path.read_bytes() for path in expected}, expected
            )

    def test_promotion_failure_restores_every_preexisting_output(self):
        real_replace = os.replace
        for fail_at_call in range(1, 5):
            with self.subTest(fail_at_call=fail_at_call):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    staged = [root / f"staged-{index}" for index in range(2)]
                    final = [root / f"final-{index}" for index in range(2)]
                    for index, path in enumerate(staged):
                        path.write_bytes(f"new-{index}".encode("ascii"))
                    for index, path in enumerate(final):
                        path.write_bytes(f"old-{index}".encode("ascii"))
                    call_count = 0

                    def fail_one_replace(source, destination):
                        nonlocal call_count
                        call_count += 1
                        if call_count == fail_at_call:
                            raise OSError(f"forced replace failure {fail_at_call}")
                        return real_replace(source, destination)

                    with mock.patch(
                        "scripts.build_lusso2020_flux_error_model.os.replace",
                        side_effect=fail_one_replace,
                    ):
                        with self.assertRaisesRegex(OSError, "forced replace"):
                            _promote_staged_outputs(zip(staged, final))

                    self.assertEqual(final[0].read_bytes(), b"old-0")
                    self.assertEqual(final[1].read_bytes(), b"old-1")

    def test_promotion_failure_removes_outputs_that_were_initially_absent(self):
        real_replace = os.replace
        for fail_at_call in range(1, 3):
            with self.subTest(fail_at_call=fail_at_call):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    staged = [root / f"staged-{index}" for index in range(2)]
                    final = [root / f"final-{index}" for index in range(2)]
                    for index, path in enumerate(staged):
                        path.write_bytes(f"new-{index}".encode("ascii"))
                    call_count = 0

                    def fail_one_replace(source, destination):
                        nonlocal call_count
                        call_count += 1
                        if call_count == fail_at_call:
                            raise OSError(f"forced replace failure {fail_at_call}")
                        return real_replace(source, destination)

                    with mock.patch(
                        "scripts.build_lusso2020_flux_error_model.os.replace",
                        side_effect=fail_one_replace,
                    ):
                        with self.assertRaisesRegex(OSError, "forced replace"):
                            _promote_staged_outputs(zip(staged, final))

                    self.assertFalse(final[0].exists())
                    self.assertFalse(final[1].exists())

    def test_no_overwrite_promotion_cannot_clobber_a_late_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged = [root / f"staged-{index}" for index in range(2)]
            final = [root / f"final-{index}" for index in range(2)]
            for index, path in enumerate(staged):
                path.write_bytes(f"new-{index}".encode("ascii"))
            final[1].write_bytes(b"appeared-during-staging")

            with self.assertRaises(FileExistsError):
                _promote_staged_outputs(
                    zip(staged, final),
                    overwrite=False,
                )

            self.assertFalse(final[0].exists())
            self.assertEqual(final[1].read_bytes(), b"appeared-during-staging")

    def test_concurrent_no_overwrite_clis_cannot_both_succeed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            figures = Path(temporary) / "figures"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "build_lusso2020_flux_error_model.py"),
                "--source",
                str(FIXTURE),
                "--output-directory",
                str(output),
                "--figures-directory",
                str(figures),
                "--retrieval-date",
                "2026-07-29",
            ]
            first_process = subprocess.Popen(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            second_process = subprocess.Popen(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            first_stdout, first_stderr = first_process.communicate(timeout=30)
            second_stdout, second_stderr = second_process.communicate(timeout=30)

        return_codes = [first_process.returncode, second_process.returncode]
        self.assertEqual(return_codes.count(0), 1)
        self.assertEqual(
            sum("--overwrite" in stderr for stderr in (first_stderr, second_stderr)),
            1,
        )
        self.assertEqual(
            sum(
                "valid rows: 12" in stdout
                for stdout in (first_stdout, second_stdout)
            ),
            1,
        )

    def test_broad_bin_checks_exclude_exact_bin_boundaries(self):
        source = source_from(
            [0.0, 0.5, 1.0, 1.5, 2.0],
            [9.0, 0.01, 8.0, 0.02, 7.0],
            [9.0, 0.04, 8.0, 0.08, 7.0],
        )

        checks = _broad_bin_checks(source)

        self.assertIn("0 < z < 1: median e_logFUV=0.010000 dex", checks[0])
        self.assertIn("(N=1)", checks[0])
        self.assertIn("1 < z < 2: median e_logFUV=0.020000 dex", checks[1])
        self.assertIn("(N=1)", checks[1])

    def test_cli_reports_outputs_and_requires_overwrite_for_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            figures = Path(temporary) / "figures"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "build_lusso2020_flux_error_model.py"),
                "--source",
                str(FIXTURE),
                "--output-directory",
                str(output),
                "--figures-directory",
                str(figures),
                "--retrieval-date",
                "2026-07-29",
            ]
            first = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False
            )
            second = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False
            )

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("valid rows: 12", first.stdout)
        self.assertIn("source SHA-256:", first.stdout)
        self.assertIn("CSV SHA-256:", first.stdout)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("--overwrite", second.stderr)


if __name__ == "__main__":
    unittest.main()
