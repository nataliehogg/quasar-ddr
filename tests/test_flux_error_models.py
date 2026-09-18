import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools import flux_error_models
from tools.flux_error_models import (
    interpolate_flux_error_table,
    load_flux_error_table,
    propagate_flux_uncertainties,
    realize_flux_uncertainties,
)


FIXTURE_CSV = (
    Path(__file__).parent / "fixtures" / "lusso2020_flux_error_model.csv"
)


class FluxErrorTableTests(unittest.TestCase):
    def _copied_fixture(self):
        temporary_directory = tempfile.TemporaryDirectory()
        directory = Path(temporary_directory.name)
        csv_path = directory / FIXTURE_CSV.name
        metadata_path = directory / "lusso2020_flux_error_model_metadata.json"
        shutil.copyfile(FIXTURE_CSV, csv_path)
        shutil.copyfile(
            FIXTURE_CSV.with_name("lusso2020_flux_error_model_metadata.json"),
            metadata_path,
        )
        self.addCleanup(temporary_directory.cleanup)
        return csv_path, metadata_path

    @staticmethod
    def _refresh_digest(csv_path, metadata_path):
        metadata = json.loads(metadata_path.read_text())
        metadata["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata))

    def test_loads_immutable_table_and_recovers_grid_nodes(self):
        table = load_flux_error_table(FIXTURE_CSV)
        self.assertEqual(table.schema_version, 1)
        self.assertFalse(table.z.flags.writeable)
        np.testing.assert_array_equal(table.z, [0.5, 1.0, 1.5])

        result = interpolate_flux_error_table(table, table.z)
        np.testing.assert_array_equal(result.redshift, table.z)
        np.testing.assert_array_equal(
            result.median_sigma_log_fuv_dex, table.median_sigma_log_fuv_dex
        )
        np.testing.assert_array_equal(
            result.median_sigma_log_fx_dex, table.median_sigma_log_fx_dex
        )

    def test_interpolation_uses_log_medians(self):
        table = load_flux_error_table(FIXTURE_CSV)
        result = interpolate_flux_error_table(table, np.array([0.75]))
        expected = np.sqrt(
            table.median_sigma_log_fx_dex[0]
            * table.median_sigma_log_fx_dex[1]
        )
        self.assertAlmostEqual(result.median_sigma_log_fx_dex[0], expected)

    def test_interpolation_is_linear_for_non_median_fields(self):
        table = load_flux_error_table(FIXTURE_CSV)
        result = interpolate_flux_error_table(table, np.array([0.75]))
        self.assertAlmostEqual(result.scatter_ln_sigma_log_fuv[0], 0.30)
        self.assertAlmostEqual(result.scatter_ln_sigma_log_fx[0], 0.40)
        self.assertAlmostEqual(result.corr_ln_uncertainty_sizes[0], -0.125)
        self.assertAlmostEqual(result.effective_reference_count[0], 52.0)
        self.assertAlmostEqual(result.smoothing_width_z[0], 0.075)

    def test_interpolation_preserves_input_order_and_shape(self):
        table = load_flux_error_table(FIXTURE_CSV)
        redshift = np.array([[1.25, 0.75], [1.5, 0.5]])
        result = interpolate_flux_error_table(table, redshift)
        self.assertEqual(result.redshift.shape, redshift.shape)
        np.testing.assert_array_equal(result.redshift, redshift)
        self.assertAlmostEqual(result.median_sigma_log_fuv_dex[0, 0], 0.02 * np.sqrt(2))

    def test_rejects_digest_mismatch(self):
        csv_path, _ = self._copied_fixture()
        csv_path.write_text(csv_path.read_text().replace("0.0100", "0.0101"))
        with self.assertRaises(ValueError):
            load_flux_error_table(csv_path)

    def test_parses_the_same_csv_bytes_that_were_digest_validated(self):
        csv_path, _ = self._copied_fixture()
        original_bytes = csv_path.read_bytes()
        replacement_bytes = original_bytes.replace(b"0.0100", b"0.0199", 1)
        original_validate = flux_error_models._validated_metadata

        def replace_path_after_validation(metadata_path, csv_sha256):
            metadata = original_validate(metadata_path, csv_sha256)
            csv_path.write_bytes(replacement_bytes)
            return metadata

        with mock.patch.object(
            flux_error_models,
            "_validated_metadata",
            side_effect=replace_path_after_validation,
        ):
            table = load_flux_error_table(csv_path)

        self.assertEqual(csv_path.read_bytes(), replacement_bytes)
        self.assertEqual(
            table.csv_sha256,
            hashlib.sha256(original_bytes).hexdigest(),
        )
        np.testing.assert_array_equal(
            table.median_sigma_log_fuv_dex,
            [0.01, 0.02, 0.04],
        )

    def test_rejects_missing_or_extra_csv_columns(self):
        for replacement in (
            "z,median_sigma_log_fuv_dex,scatter_ln_sigma_log_fuv,median_sigma_log_fx_dex,scatter_ln_sigma_log_fx,corr_ln_uncertainty_sizes,effective_reference_count",
            "z,median_sigma_log_fuv_dex,scatter_ln_sigma_log_fuv,median_sigma_log_fx_dex,scatter_ln_sigma_log_fx,corr_ln_uncertainty_sizes,effective_reference_count,smoothing_width_z,unexpected",
        ):
            csv_path, metadata_path = self._copied_fixture()
            lines = csv_path.read_text().splitlines()
            csv_path.write_text("\n".join([replacement, *lines[1:]]) + "\n")
            self._refresh_digest(csv_path, metadata_path)
            with self.assertRaises(ValueError):
                load_flux_error_table(csv_path)

    def test_rejects_surplus_cells_in_a_digest_matched_csv_row(self):
        csv_path, metadata_path = self._copied_fixture()
        lines = csv_path.read_text().splitlines()
        lines[1] = f"{lines[1]},surplus"
        csv_path.write_text("\n".join(lines) + "\n")
        self._refresh_digest(csv_path, metadata_path)
        with self.assertRaises(ValueError):
            load_flux_error_table(csv_path)

    def test_rejects_metadata_support_endpoint_mismatch_beyond_serialization_noise(self):
        csv_path, metadata_path = self._copied_fixture()
        metadata = json.loads(metadata_path.read_text())
        metadata["supported_redshift_min"] = 0.500004
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            load_flux_error_table(csv_path)

    def test_rejects_non_monotonic_nodes(self):
        csv_path, metadata_path = self._copied_fixture()
        lines = csv_path.read_text().splitlines()
        csv_path.write_text("\n".join([lines[0], lines[1], lines[3], lines[2]]) + "\n")
        self._refresh_digest(csv_path, metadata_path)
        with self.assertRaises(ValueError):
            load_flux_error_table(csv_path)

    def test_rejects_non_positive_medians(self):
        csv_path, metadata_path = self._copied_fixture()
        csv_path.write_text(csv_path.read_text().replace("0.0400", "0.0000"))
        self._refresh_digest(csv_path, metadata_path)
        with self.assertRaises(ValueError):
            load_flux_error_table(csv_path)

    def test_rejects_negative_scatters(self):
        csv_path, metadata_path = self._copied_fixture()
        csv_path.write_text(csv_path.read_text().replace("0.20", "-0.20", 1))
        self._refresh_digest(csv_path, metadata_path)
        with self.assertRaises(ValueError):
            load_flux_error_table(csv_path)

    def test_rejects_invalid_size_correlations(self):
        csv_path, metadata_path = self._copied_fixture()
        csv_path.write_text(csv_path.read_text().replace("0.25", "1.01"))
        self._refresh_digest(csv_path, metadata_path)
        with self.assertRaises(ValueError):
            load_flux_error_table(csv_path)

    def test_rejects_non_positive_support_diagnostics(self):
        csv_path, metadata_path = self._copied_fixture()
        csv_path.write_text(csv_path.read_text().replace("48.0", "0.0"))
        self._refresh_digest(csv_path, metadata_path)
        with self.assertRaises(ValueError):
            load_flux_error_table(csv_path)

    def test_rejects_mock_redshifts_outside_table_support(self):
        table = load_flux_error_table(FIXTURE_CSV)
        with self.assertRaises(ValueError):
            interpolate_flux_error_table(table, np.array([0.49, 1.0]))
        with self.assertRaises(ValueError):
            interpolate_flux_error_table(table, np.array([1.51]))


class FluxPropagationTests(unittest.TestCase):
    def test_reference_propagation_matches_work_notes(self):
        result = propagate_flux_uncertainties(
            sigma_log_fuv_dex=np.array([0.0101]),
            sigma_log_fx_dex=np.array([0.0451]),
            gamma=0.60,
            delta_dex=0.21,
            flux_measurement_error_correlation=0.0,
        )
        self.assertAlmostEqual(result.sigma_uvx_total_dex[0], 0.2149, places=4)
        self.assertAlmostEqual(result.sigma_ln_dl[0], 0.6185, places=4)

    def test_gamma_changes_propagated_distance_uncertainty(self):
        common = dict(
            sigma_log_fuv_dex=np.array([0.01]),
            sigma_log_fx_dex=np.array([0.05]),
            delta_dex=0.21,
            flux_measurement_error_correlation=0.0,
        )
        low_gamma = propagate_flux_uncertainties(gamma=0.40, **common)
        high_gamma = propagate_flux_uncertainties(gamma=0.70, **common)
        self.assertGreater(high_gamma.sigma_ln_dl[0], low_gamma.sigma_ln_dl[0])

    def test_delta_changes_only_intrinsic_component(self):
        common = dict(
            sigma_log_fuv_dex=np.array([0.01]),
            sigma_log_fx_dex=np.array([0.05]),
            gamma=0.60,
            flux_measurement_error_correlation=0.0,
        )
        small = propagate_flux_uncertainties(delta_dex=0.15, **common)
        large = propagate_flux_uncertainties(delta_dex=0.25, **common)
        np.testing.assert_array_equal(
            small.sigma_uvx_measurement_dex, large.sigma_uvx_measurement_dex
        )
        self.assertGreater(
            large.sigma_uvx_intrinsic_dex[0], small.sigma_uvx_intrinsic_dex[0]
        )

    def test_signed_measurement_error_correlation_enters_covariance_term(self):
        common = dict(
            sigma_log_fuv_dex=np.array([0.01]),
            sigma_log_fx_dex=np.array([0.05]),
            gamma=0.60,
            delta_dex=0.21,
        )
        positive = propagate_flux_uncertainties(
            flux_measurement_error_correlation=1.0, **common
        )
        negative = propagate_flux_uncertainties(
            flux_measurement_error_correlation=-1.0, **common
        )
        self.assertAlmostEqual(positive.sigma_uvx_measurement_dex[0], 0.044)
        self.assertAlmostEqual(negative.sigma_uvx_measurement_dex[0], 0.056)

    def test_zero_measurement_errors_leave_intrinsic_dispersion(self):
        result = propagate_flux_uncertainties(
            sigma_log_fuv_dex=np.array([0.0]),
            sigma_log_fx_dex=np.array([0.0]),
            gamma=0.60,
            delta_dex=0.21,
            flux_measurement_error_correlation=0.0,
        )
        self.assertEqual(result.sigma_uvx_measurement_dex[0], 0.0)
        self.assertEqual(result.sigma_uvx_intrinsic_dex[0], 0.21)
        self.assertEqual(result.sigma_uvx_total_dex[0], 0.21)

    def test_zero_intrinsic_dispersion_permits_measurement_only_propagation(self):
        result = propagate_flux_uncertainties(
            sigma_log_fuv_dex=np.array([0.01]),
            sigma_log_fx_dex=np.array([0.05]),
            gamma=0.60,
            delta_dex=0.0,
            flux_measurement_error_correlation=0.0,
        )
        self.assertEqual(result.sigma_uvx_intrinsic_dex[0], 0.0)
        self.assertAlmostEqual(result.sigma_uvx_total_dex[0], result.sigma_uvx_measurement_dex[0])

    def test_rejects_invalid_measurement_arrays_and_model_parameters(self):
        valid = dict(
            sigma_log_fuv_dex=np.array([0.01]),
            sigma_log_fx_dex=np.array([0.05]),
            gamma=0.60,
            delta_dex=0.21,
            flux_measurement_error_correlation=0.0,
        )
        invalid_cases = (
            {"sigma_log_fx_dex": np.array([0.05, 0.06])},
            {"sigma_log_fuv_dex": np.array([-0.01])},
            {"sigma_log_fx_dex": np.array([np.inf])},
            {"gamma": -0.01},
            {"gamma": 1.0},
            {"delta_dex": -0.1},
            {"flux_measurement_error_correlation": 1.01},
        )
        for update in invalid_cases:
            with self.subTest(update=update), self.assertRaises(ValueError):
                propagate_flux_uncertainties(**{**valid, **update})

    def test_roundoff_only_negative_measurement_radicand_is_clipped_to_zero(self):
        result = propagate_flux_uncertainties(
            sigma_log_fuv_dex=np.array([0.1]),
            sigma_log_fx_dex=np.array([0.06]),
            gamma=0.60,
            delta_dex=0.21,
            flux_measurement_error_correlation=1.0,
        )
        self.assertGreaterEqual(result.sigma_uvx_measurement_dex[0], 0.0)


class FluxUncertaintyRealizationTests(unittest.TestCase):
    def setUp(self):
        self.table = load_flux_error_table(FIXTURE_CSV)
        self.parameters = dict(
            gamma=0.60,
            delta_dex=0.21,
            flux_measurement_error_correlation=0.0,
        )

    def test_random_draws_are_strictly_positive_and_continuous(self):
        result = realize_flux_uncertainties(
            self.table,
            np.full(100, 0.75),
            mode="random",
            rng=np.random.default_rng(20260729),
            **self.parameters,
        )
        self.assertTrue(np.all(result.sigma_log_fuv_dex > 0))
        self.assertTrue(np.all(result.sigma_log_fx_dex > 0))
        self.assertGreater(np.unique(result.sigma_log_fuv_dex).size, 90)
        self.assertGreater(np.unique(result.sigma_log_fx_dex).size, 90)

    def test_random_draws_are_exactly_reproducible_for_a_seed(self):
        redshift = np.array([0.5, 0.75, 1.25])
        first = realize_flux_uncertainties(
            self.table, redshift, mode="random", rng=np.random.default_rng(17), **self.parameters
        )
        second = realize_flux_uncertainties(
            self.table, redshift, mode="random", rng=np.random.default_rng(17), **self.parameters
        )
        np.testing.assert_array_equal(first.sigma_log_fuv_dex, second.sigma_log_fuv_dex)
        np.testing.assert_array_equal(first.sigma_log_fx_dex, second.sigma_log_fx_dex)

    def test_random_draws_change_for_a_different_seed(self):
        redshift = np.full(20, 0.75)
        first = realize_flux_uncertainties(
            self.table, redshift, mode="random", rng=np.random.default_rng(17), **self.parameters
        )
        second = realize_flux_uncertainties(
            self.table, redshift, mode="random", rng=np.random.default_rng(18), **self.parameters
        )
        self.assertFalse(
            np.array_equal(first.sigma_log_fuv_dex, second.sigma_log_fuv_dex)
        )

    def test_random_mode_requires_a_numpy_generator(self):
        class StandardNormalLookalike:
            def standard_normal(self, shape):
                return np.zeros(shape)

        for invalid_rng in (np.random.RandomState(17), StandardNormalLookalike()):
            with self.subTest(invalid_rng=type(invalid_rng).__name__), self.assertRaises(
                ValueError
            ):
                realize_flux_uncertainties(
                    self.table,
                    np.array([0.75]),
                    mode="random",
                    rng=invalid_rng,
                    **self.parameters,
                )

    def test_random_draws_recover_interpolated_log_space_distribution(self):
        count = 50000
        redshift = np.full(count, 0.75)
        expected = interpolate_flux_error_table(self.table, redshift)
        result = realize_flux_uncertainties(
            self.table,
            redshift,
            mode="random",
            rng=np.random.default_rng(461),
            **self.parameters,
        )
        log_uv = np.log(result.sigma_log_fuv_dex)
        log_x = np.log(result.sigma_log_fx_dex)
        self.assertAlmostEqual(
            np.median(log_uv), np.log(expected.median_sigma_log_fuv_dex[0]), places=2
        )
        self.assertAlmostEqual(
            np.median(log_x), np.log(expected.median_sigma_log_fx_dex[0]), places=2
        )
        self.assertAlmostEqual(
            np.std(log_uv), expected.scatter_ln_sigma_log_fuv[0], places=2
        )
        self.assertAlmostEqual(
            np.std(log_x), expected.scatter_ln_sigma_log_fx[0], places=2
        )
        self.assertLess(
            abs(
                np.corrcoef(log_uv, log_x)[0, 1]
                - expected.corr_ln_uncertainty_sizes[0]
            ),
            0.015,
        )

    def test_asimov_uses_interpolated_medians_exactly(self):
        result = realize_flux_uncertainties(
            self.table,
            np.array([0.75]),
            mode="asimov",
            **self.parameters,
        )
        expected = interpolate_flux_error_table(self.table, np.array([0.75]))
        np.testing.assert_array_equal(
            result.sigma_log_fuv_dex,
            expected.median_sigma_log_fuv_dex,
        )
        np.testing.assert_array_equal(
            result.sigma_log_fx_dex,
            expected.median_sigma_log_fx_dex,
        )

    def test_asimov_mode_does_not_consume_a_provided_generator(self):
        expected_rng = np.random.default_rng(20260729)
        expected_draws = expected_rng.standard_normal(5)
        rng = np.random.default_rng(20260729)
        result = realize_flux_uncertainties(
            self.table,
            np.array([0.75]),
            mode="asimov",
            rng=rng,
            **self.parameters,
        )
        self.assertEqual(result.model_name, "lusso2020_flux_table")
        np.testing.assert_array_equal(rng.standard_normal(5), expected_draws)


if __name__ == "__main__":
    unittest.main()
