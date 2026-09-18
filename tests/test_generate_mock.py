import csv
from contextlib import redirect_stdout
from io import StringIO
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest import mock

import numpy as np

import generate_mock
from tools.error_models import resolve_distance_errors
from tools.flux_error_models import realize_flux_uncertainties
from tools.qso_mock import load_paired_log_mock
from tools.qso_mock import generate_paired_log_mock
from tools.theory import angular_diameter_distance, ddr_eta


FLUX_TABLE_FIXTURE = (
    Path(__file__).parent / "fixtures" / "lusso2020_flux_error_model.csv"
)


class MockConfigurationTests(unittest.TestCase):
    def test_editable_uncertainty_constants_are_copied_independently(self):
        config = generate_mock.configuration_from_constants()
        required_constants = (
            "UVX_GAMMA",
            "UVX_ERROR_OPTION",
            "UVX_DELTA_DEX",
            "FLUX_MEASUREMENT_ERROR_CORRELATION",
            "FLUX_ERROR_TABLE",
            "SARM_ERROR_OPTION",
        )
        self.assertEqual(
            [name for name in required_constants if not hasattr(generate_mock, name)],
            [],
        )
        expected = {
            "uvx_gamma": generate_mock.UVX_GAMMA,
            "uvx_error_option": generate_mock.UVX_ERROR_OPTION,
            "custom_sigma_ln_dl": generate_mock.CUSTOM_SIGMA_LN_DL,
            "uvx_delta_dex": generate_mock.UVX_DELTA_DEX,
            "flux_measurement_error_correlation": (
                generate_mock.FLUX_MEASUREMENT_ERROR_CORRELATION
            ),
            "flux_error_table": generate_mock.FLUX_ERROR_TABLE,
            "sarm_error_option": generate_mock.SARM_ERROR_OPTION,
            "custom_sigma_ln_da": generate_mock.CUSTOM_SIGMA_LN_DA,
            "rho": generate_mock.RHO,
        }
        self.assertEqual(
            {name: getattr(config, name) for name in expected},
            expected,
        )
        self.assertNotIn(
            "error_scenario",
            {field.name for field in fields(generate_mock.MockConfiguration)},
        )
        self.assertFalse(hasattr(generate_mock, "ERROR_SCENARIO"))

    def test_documented_uvx_options_are_accepted_in_exact_order(self):
        self.assertTrue(
            hasattr(generate_mock, "ValidatedErrorConfiguration"),
            "validation must return the cached independent error configuration",
        )
        options = (
            "current",
            "future",
            "targeted",
            "custom",
            "lusso2020_flux_table",
        )
        for option in options:
            with self.subTest(option=option):
                config = generate_mock.MockConfiguration(
                    z_min=0.5,
                    z_max=1.5,
                    uvx_error_option=option,
                    flux_error_table=str(FLUX_TABLE_FIXTURE),
                )
                validated = generate_mock.validate_configuration(config)
                if option == "lusso2020_flux_table":
                    self.assertIsNone(validated.uvx)
                    self.assertIsNotNone(validated.flux_table)
                else:
                    self.assertEqual(validated.uvx.option, option)
                    self.assertIsNone(validated.flux_table)

        with self.assertRaisesRegex(
            KeyError,
            "current, future, targeted, custom, lusso2020_flux_table",
        ):
            generate_mock.validate_configuration(
                generate_mock.MockConfiguration(uvx_error_option="aggressive")
            )

    def test_documented_sarm_options_are_accepted_in_exact_order(self):
        options = ("current", "future", "targeted", "custom")
        for option in options:
            with self.subTest(option=option):
                validated = generate_mock.validate_configuration(
                    generate_mock.MockConfiguration(sarm_error_option=option)
                )
                self.assertEqual(validated.sarm.option, option)

        with self.assertRaisesRegex(
            KeyError,
            "current, future, targeted, custom",
        ):
            generate_mock.validate_configuration(
                generate_mock.MockConfiguration(sarm_error_option="aggressive")
            )

    def test_gamma_must_satisfy_closed_open_unit_interval(self):
        for invalid_gamma in (-0.01, 1.0, np.nan, np.inf):
            with self.subTest(uvx_gamma=invalid_gamma):
                with self.assertRaisesRegex(ValueError, "UVX_GAMMA"):
                    generate_mock.validate_configuration(
                        generate_mock.MockConfiguration(uvx_gamma=invalid_gamma)
                    )

    def test_delta_must_be_finite_and_non_negative_even_when_inactive(self):
        for invalid_delta in (-0.01, np.nan, np.inf):
            with self.subTest(uvx_delta_dex=invalid_delta):
                with self.assertRaisesRegex(ValueError, "UVX_DELTA_DEX"):
                    generate_mock.validate_configuration(
                        generate_mock.MockConfiguration(
                            uvx_error_option="current",
                            uvx_delta_dex=invalid_delta,
                        )
                    )

    def test_both_correlations_must_be_finite_and_within_unit_bounds(self):
        cases = (
            ("flux_measurement_error_correlation", -1.01),
            ("flux_measurement_error_correlation", 1.01),
            ("flux_measurement_error_correlation", np.nan),
            ("flux_measurement_error_correlation", 0.5j),
            ("flux_measurement_error_correlation", 0.5 + 0j),
            ("rho", -1.01),
            ("rho", 1.01),
            ("rho", np.nan),
            ("rho", 0.5j),
            ("rho", 0.5 + 0j),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field.upper()):
                    generate_mock.validate_configuration(
                        generate_mock.MockConfiguration(**{field: value})
                    )

    def test_custom_values_must_be_valid_even_when_inactive(self):
        for field in ("custom_sigma_ln_dl", "custom_sigma_ln_da"):
            for value in (-0.01, np.nan, np.inf):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field.upper()):
                        generate_mock.validate_configuration(
                            generate_mock.MockConfiguration(
                                uvx_error_option="current",
                                sarm_error_option="current",
                                **{field: value},
                            )
                        )

    def test_inactive_table_path_is_not_loaded(self):
        missing = Path(tempfile.gettempdir()) / "task5_missing_inactive_table.csv"
        validated = generate_mock.validate_configuration(
            generate_mock.MockConfiguration(
                uvx_error_option="current",
                flux_error_table=str(missing),
            )
        )
        self.assertIsNone(validated.flux_table)

    def test_table_option_rejects_missing_and_invalid_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            with self.assertRaisesRegex(ValueError, "flux-error table"):
                generate_mock.validate_configuration(
                    generate_mock.MockConfiguration(
                        z_min=0.5,
                        z_max=1.5,
                        uvx_error_option="lusso2020_flux_table",
                        flux_error_table=str(missing),
                    )
                )

            invalid = Path(directory) / FLUX_TABLE_FIXTURE.name
            metadata = invalid.with_name(
                "lusso2020_flux_error_model_metadata.json"
            )
            shutil.copyfile(FLUX_TABLE_FIXTURE, invalid)
            shutil.copyfile(
                FLUX_TABLE_FIXTURE.with_name(
                    "lusso2020_flux_error_model_metadata.json"
                ),
                metadata,
            )
            invalid.write_text(invalid.read_text().replace("0.0100", "0.0101"))
            with self.assertRaisesRegex(ValueError, "digest"):
                generate_mock.validate_configuration(
                    generate_mock.MockConfiguration(
                        z_min=0.5,
                        z_max=1.5,
                        uvx_error_option="lusso2020_flux_table",
                        flux_error_table=str(invalid),
                    )
                )

    def test_table_option_rejects_requested_redshift_range_outside_support(self):
        for z_min, z_max in ((0.49, 1.5), (0.5, 1.51)):
            with self.subTest(z_min=z_min, z_max=z_max):
                with self.assertRaisesRegex(
                    ValueError,
                    "requested.*outside.*supported",
                ):
                    generate_mock.validate_configuration(
                        generate_mock.MockConfiguration(
                            z_min=z_min,
                            z_max=z_max,
                            uvx_error_option="lusso2020_flux_table",
                            flux_error_table=str(FLUX_TABLE_FIXTURE),
                        )
                    )

    def test_relative_table_path_is_resolved_from_generator_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            original_cwd = Path.cwd()
            try:
                os.chdir(directory)
                validated = generate_mock.validate_configuration(
                    generate_mock.MockConfiguration(
                        uvx_error_option="lusso2020_flux_table",
                    )
                )
            finally:
                os.chdir(original_cwd)

        expected = (
            Path(generate_mock.__file__).resolve().parent
            / generate_mock.FLUX_ERROR_TABLE
        )
        self.assertEqual(validated.flux_table.csv_path, expected)

    def test_generator_import_and_generation_do_not_require_cobaya(self):
        """A fitting import would fail in this deliberately Cobaya-free process."""
        repository = Path(__file__).resolve().parents[1]
        script = textwrap.dedent(
            """
            import builtins

            real_import = builtins.__import__

            def cobaya_free_import(name, *args, **kwargs):
                if name == "cobaya" or name.startswith("cobaya."):
                    raise ModuleNotFoundError("Cobaya is deliberately unavailable")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = cobaya_free_import

            import generate_mock

            config = generate_mock.MockConfiguration(
                n_qso=3,
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
            generated = generate_mock.build_catalogue(config)
            assert generated.mock.redshift.size == 3
            assert (generated.mock.dl_observed == generated.mock.dl_true).all()
            """
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository) + os.pathsep + environment.get(
            "PYTHONPATH", ""
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tempfile.gettempdir(),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_configuration_names_the_setting(self):
        config = generate_mock.MockConfiguration(n_qso=0)
        with self.assertRaisesRegex(ValueError, "N_QSO"):
            generate_mock.validate_configuration(config)

    def test_every_numeric_setting_must_be_finite_even_when_inactive(self):
        cases = (
            ("n_qso", np.nan),
            ("z_min", np.nan),
            ("z_max", np.inf),
            ("h0", np.nan),
            ("omega_m", np.inf),
            ("omega_k", np.nan),
            ("w0", np.inf),
            ("alpha", np.inf),
            ("beta", np.nan),
            ("custom_sigma_ln_dl", np.nan),
            ("custom_sigma_ln_da", np.inf),
            ("rho", np.nan),
        )
        for field, invalid_value in cases:
            with self.subTest(field=field):
                config = generate_mock.MockConfiguration(
                    mock_mode="asimov",
                    make_diagnostic_plot=False,
                    **{field: invalid_value},
                )
                with self.assertRaises(ValueError):
                    generate_mock.validate_configuration(config)

    def test_random_seed_rejects_invalid_values_even_for_asimov_mocks(self):
        for invalid_seed in (-1, np.int64(-1), 1.5, True):
            with self.subTest(random_seed=invalid_seed):
                config = generate_mock.MockConfiguration(
                    random_seed=invalid_seed,
                    mock_mode="asimov",
                    make_diagnostic_plot=False,
                )
                with self.assertRaisesRegex(ValueError, "RANDOM_SEED"):
                    generate_mock.validate_configuration(config)

    def test_numpy_integer_seed_builds_and_serializes_as_an_integer(self):
        with tempfile.TemporaryDirectory() as directory:
            config = generate_mock.MockConfiguration(
                n_qso=np.int64(3),
                random_seed=np.int64(23),
                output_directory=directory,
                make_diagnostic_plot=False,
            )
            generated = generate_mock.build_catalogue(config)
            paths = generate_mock.write_catalogue(config, generated)
            metadata = json.loads(paths["metadata"].read_text())

            self.assertEqual(metadata["configuration"]["n_qso"], 3)
            self.assertEqual(metadata["configuration"]["random_seed"], 23)
            self.assertEqual(metadata["mock_generation"]["master_seed"], 23)

    def test_seed_above_npz_int64_range_fails_before_output_changes(self):
        invalid_seeds = (2**100, np.uint64(2**63))
        for invalid_seed in invalid_seeds:
            with self.subTest(random_seed=invalid_seed):
                with tempfile.TemporaryDirectory() as directory:
                    output_directory = Path(directory) / "catalogue"
                    config = generate_mock.MockConfiguration(
                        n_qso=3,
                        random_seed=invalid_seed,
                        output_directory=str(output_directory),
                        make_diagnostic_plot=False,
                        overwrite=True,
                    )
                    output_directory.mkdir()
                    csv_path = (
                        output_directory
                        / f"{generate_mock.dataset_stem(config)}.csv"
                    )
                    csv_path.write_bytes(b"existing catalogue")

                    error = None
                    try:
                        generated = generate_mock.build_catalogue(config)
                        generate_mock.write_catalogue(config, generated)
                    except Exception as exc:
                        error = exc

                    self.assertEqual(csv_path.read_bytes(), b"existing catalogue")
                    self.assertEqual(
                        {path.name for path in output_directory.iterdir()},
                        {csv_path.name},
                    )
                    self.assertIsInstance(error, ValueError)
                    self.assertRegex(str(error), "RANDOM_SEED")


class CatalogueTests(unittest.TestCase):
    def test_generated_catalogue_exposes_independent_error_models(self):
        generated = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=3,
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
        )

        self.assertEqual(
            [field.name for field in fields(generate_mock.GeneratedCatalogue)],
            [
                "mock",
                "eta_true",
                "uvx_errors",
                "sarm_errors",
                "flux_uncertainties",
                "flux_table",
            ],
        )
        self.assertEqual(generated.uvx_errors.option, "current")
        self.assertEqual(generated.sarm_errors.option, "current")
        self.assertIsNone(generated.flux_uncertainties)
        self.assertIsNone(generated.flux_table)

    def test_matched_named_choices_reproduce_legacy_catalogues_at_gamma_point_60(self):
        for option in ("current", "future", "targeted"):
            with self.subTest(option=option):
                config = generate_mock.MockConfiguration(
                    n_qso=7,
                    z_min=0.2,
                    z_max=1.0,
                    uvx_gamma=0.60,
                    uvx_error_option=option,
                    sarm_error_option=option,
                    random_seed=29,
                    make_diagnostic_plot=False,
                )
                generated = generate_mock.build_catalogue(config)
                redshift_stream, noise_stream = np.random.SeedSequence(
                    config.random_seed
                ).spawn(2)
                redshift = generate_mock.QSOSpecs.redshift_distribution(
                    {
                        "n_QSO": config.n_qso,
                        "z_min": config.z_min,
                        "z_max": config.z_max,
                    },
                    rng=np.random.default_rng(redshift_stream),
                    mode="random",
                )
                da_true = angular_diameter_distance(
                    redshift,
                    config.h0,
                    config.omega_m,
                    config.cosmology_model,
                    omega_k=config.omega_k,
                    w0=config.w0,
                )
                eta_true = ddr_eta(
                    redshift,
                    config.ddr_model,
                    config.alpha,
                    config.beta,
                )
                legacy_errors = resolve_distance_errors(option)
                legacy = generate_paired_log_mock(
                    redshift,
                    da_true,
                    eta_true,
                    legacy_errors.sigma_ln_dl,
                    legacy_errors.sigma_ln_da,
                    rho=config.rho,
                    mode="random",
                    seed=config.random_seed,
                    rng=np.random.default_rng(noise_stream),
                    errors_include_intrinsic=True,
                )

                for field in fields(legacy):
                    actual = getattr(generated.mock, field.name)
                    expected = getattr(legacy, field.name)
                    if isinstance(expected, np.ndarray):
                        np.testing.assert_array_equal(actual, expected)
                    else:
                        self.assertEqual(actual, expected)

    def test_mixed_uvx_and_sarm_choices_resolve_independently(self):
        generated = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=4,
                uvx_error_option="future",
                sarm_error_option="targeted",
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
        )

        self.assertEqual(generated.uvx_errors.option, "future")
        self.assertEqual(generated.sarm_errors.option, "targeted")
        np.testing.assert_allclose(
            generated.mock.sigma_ln_dl,
            np.log(10.0) * 0.17 / (2.0 * (1.0 - 0.60)),
        )
        np.testing.assert_allclose(generated.mock.sigma_ln_da, 0.15)

    def test_custom_uvx_distance_error_is_invariant_under_gamma_changes(self):
        low_gamma = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=4,
                uvx_gamma=0.2,
                uvx_error_option="custom",
                custom_sigma_ln_dl=0.31,
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
        )
        high_gamma = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=4,
                uvx_gamma=0.8,
                uvx_error_option="custom",
                custom_sigma_ln_dl=0.31,
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
        )

        np.testing.assert_array_equal(
            low_gamma.mock.sigma_ln_dl,
            high_gamma.mock.sigma_ln_dl,
        )
        np.testing.assert_array_equal(
            low_gamma.mock.dl_observed,
            high_gamma.mock.dl_observed,
        )
        self.assertFalse(low_gamma.uvx_errors.gamma_used)
        self.assertFalse(high_gamma.uvx_errors.gamma_used)

    def test_named_uvx_distance_error_changes_with_gamma(self):
        low_gamma = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=4,
                uvx_gamma=0.5,
                uvx_error_option="current",
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
        )
        high_gamma = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=4,
                uvx_gamma=0.7,
                uvx_error_option="current",
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
        )

        np.testing.assert_allclose(
            low_gamma.mock.sigma_ln_dl,
            np.log(10.0) * 0.24 / (2.0 * (1.0 - 0.5)),
        )
        np.testing.assert_allclose(
            high_gamma.mock.sigma_ln_dl,
            np.log(10.0) * 0.24 / (2.0 * (1.0 - 0.7)),
        )
        self.assertFalse(
            np.array_equal(
                low_gamma.mock.sigma_ln_dl,
                high_gamma.mock.sigma_ln_dl,
            )
        )

    def test_table_mode_uses_child_2_for_per_object_flux_uncertainties(self):
        config = generate_mock.MockConfiguration(
            n_qso=8,
            z_min=0.5,
            z_max=1.5,
            uvx_error_option="lusso2020_flux_table",
            flux_error_table=str(FLUX_TABLE_FIXTURE),
            random_seed=41,
            make_diagnostic_plot=False,
        )
        generated = generate_mock.build_catalogue(config)
        children = np.random.SeedSequence(config.random_seed).spawn(3)
        validated = generate_mock.validate_configuration(config)
        expected = realize_flux_uncertainties(
            validated.flux_table,
            generated.mock.redshift,
            gamma=config.uvx_gamma,
            delta_dex=config.uvx_delta_dex,
            flux_measurement_error_correlation=(
                config.flux_measurement_error_correlation
            ),
            mode="random",
            rng=np.random.default_rng(children[2]),
        )

        self.assertIsNone(generated.uvx_errors)
        self.assertIsNotNone(generated.flux_uncertainties)
        for field in fields(expected):
            actual_value = getattr(generated.flux_uncertainties, field.name)
            expected_value = getattr(expected, field.name)
            if isinstance(expected_value, np.ndarray):
                np.testing.assert_array_equal(actual_value, expected_value)
            else:
                self.assertEqual(actual_value, expected_value)
        np.testing.assert_array_equal(
            generated.mock.sigma_ln_dl,
            expected.sigma_ln_dl,
        )
        self.assertEqual(generated.mock.sigma_ln_dl.shape, (config.n_qso,))
        self.assertGreater(np.ptp(generated.mock.sigma_ln_dl), 0.0)

    def test_changing_only_uvx_option_leaves_redshifts_and_sarm_draws_unchanged(self):
        common = dict(
            n_qso=8,
            z_min=0.5,
            z_max=1.5,
            sarm_error_option="future",
            rho=0.35,
            random_seed=47,
            make_diagnostic_plot=False,
        )
        effective = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                **common,
                uvx_error_option="current",
            )
        )
        table = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                **common,
                uvx_error_option="lusso2020_flux_table",
                flux_error_table=str(FLUX_TABLE_FIXTURE),
            )
        )

        np.testing.assert_array_equal(
            effective.mock.redshift,
            table.mock.redshift,
        )
        np.testing.assert_array_equal(
            effective.mock.sigma_ln_da,
            table.mock.sigma_ln_da,
        )
        np.testing.assert_array_equal(
            effective.mock.da_observed,
            table.mock.da_observed,
        )
        self.assertEqual(effective.sarm_errors, table.sarm_errors)

    def test_effective_and_table_error_models_are_never_combined(self):
        effective = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=4,
                z_min=0.5,
                z_max=1.5,
                uvx_error_option="future",
                make_diagnostic_plot=False,
            )
        )
        table = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=4,
                z_min=0.5,
                z_max=1.5,
                uvx_error_option="lusso2020_flux_table",
                flux_error_table=str(FLUX_TABLE_FIXTURE),
                make_diagnostic_plot=False,
            )
        )

        self.assertIsNotNone(effective.uvx_errors)
        self.assertIsNone(effective.flux_uncertainties)
        self.assertIsNone(table.uvx_errors)
        self.assertIsNotNone(table.flux_uncertainties)

    def test_table_errors_are_not_coerced_into_legacy_scalar_resolved_errors(self):
        table = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=4,
                z_min=0.5,
                z_max=1.5,
                uvx_error_option="lusso2020_flux_table",
                flux_error_table=str(FLUX_TABLE_FIXTURE),
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "tabulated flux uncertainties.*object-level output",
        ):
            table.resolved_errors

    def test_asimov_table_mode_keeps_expected_object_errors_without_random_generators(
        self,
    ):
        config = generate_mock.MockConfiguration(
            n_qso=8,
            z_min=0.5,
            z_max=1.5,
            uvx_error_option="lusso2020_flux_table",
            flux_error_table=str(FLUX_TABLE_FIXTURE),
            mock_mode="asimov",
            random_seed=73,
            make_diagnostic_plot=False,
        )
        with mock.patch(
            "generate_mock.np.random.default_rng",
            side_effect=AssertionError("Asimov must not construct a generator"),
        ):
            generated = generate_mock.build_catalogue(config)

        np.testing.assert_array_equal(
            generated.mock.dl_observed,
            generated.mock.dl_true,
        )
        np.testing.assert_array_equal(
            generated.mock.da_observed,
            generated.mock.da_true,
        )
        np.testing.assert_array_equal(
            generated.mock.sigma_ln_dl,
            generated.flux_uncertainties.sigma_ln_dl,
        )
        self.assertGreater(np.ptp(generated.mock.sigma_ln_dl), 0.0)

    def test_random_catalogue_is_reproducible_and_sorted(self):
        config = generate_mock.MockConfiguration(
            n_qso=12,
            z_min=0.2,
            z_max=1.0,
            random_seed=17,
            make_diagnostic_plot=False,
        )
        first = generate_mock.build_catalogue(config)
        second = generate_mock.build_catalogue(config)
        np.testing.assert_array_equal(first.mock.redshift, second.mock.redshift)
        np.testing.assert_array_equal(
            first.mock.dl_observed, second.mock.dl_observed
        )
        self.assertTrue(np.all(np.diff(first.mock.redshift) >= 0))

    def test_different_seeds_change_redshifts_and_noise(self):
        first = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=12, random_seed=17, make_diagnostic_plot=False
            )
        )
        second = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=12, random_seed=18, make_diagnostic_plot=False
            )
        )
        self.assertFalse(np.array_equal(first.mock.redshift, second.mock.redshift))
        self.assertFalse(
            np.array_equal(first.mock.dl_observed, second.mock.dl_observed)
        )

    def test_asimov_ignores_seed_and_observed_equals_truth(self):
        first = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=8,
                mock_mode="asimov",
                random_seed=1,
                make_diagnostic_plot=False,
            )
        )
        second = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=8,
                mock_mode="asimov",
                random_seed=999,
                make_diagnostic_plot=False,
            )
        )
        np.testing.assert_array_equal(first.mock.redshift, second.mock.redshift)
        np.testing.assert_array_equal(
            first.mock.dl_observed, first.mock.dl_true
        )
        np.testing.assert_array_equal(
            first.mock.da_observed, first.mock.da_true
        )

    def test_custom_errors_are_written_into_mock(self):
        generated = generate_mock.build_catalogue(
            generate_mock.MockConfiguration(
                n_qso=5,
                uvx_error_option="custom",
                sarm_error_option="custom",
                custom_sigma_ln_dl=0.31,
                custom_sigma_ln_da=0.22,
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
        )
        np.testing.assert_allclose(generated.mock.sigma_ln_dl, 0.31)
        np.testing.assert_allclose(generated.mock.sigma_ln_da, 0.22)

    def test_build_catalogue_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "not-written"
            config = generate_mock.MockConfiguration(
                n_qso=3,
                output_directory=str(output_directory),
                make_diagnostic_plot=False,
            )
            generate_mock.build_catalogue(config)
            self.assertFalse(output_directory.exists())


class OutputTests(unittest.TestCase):
    def _printed_summary(self, config, generated, paths):
        output = StringIO()
        error = None
        try:
            with redirect_stdout(output):
                generate_mock.print_summary(config, generated, paths)
        except Exception as exc:
            error = exc
        self.assertIsNone(error)
        return output.getvalue()

    def test_named_terminal_summary_reports_independent_error_choices(self):
        config = generate_mock.MockConfiguration(
            n_qso=3,
            uvx_error_option="current",
            sarm_error_option="future",
            mock_mode="asimov",
            make_diagnostic_plot=False,
        )
        generated = generate_mock.build_catalogue(config)
        paths = {
            "csv": Path("/tmp/named.csv"),
            "metadata": Path("/tmp/named_metadata.json"),
        }

        summary = self._printed_summary(config, generated, paths)

        for text in (
            "UV-X: current",
            "gamma: 0.6 (used)",
            "residual: 0.24 dex",
            "sigma_ln_dl: 0.690776",
            "SARM: future",
            "sigma_ln_da: 0.2",
            "/tmp/named.csv",
            "/tmp/named_metadata.json",
        ):
            self.assertIn(text, summary)
        self.assertNotIn("min/median/max", summary)

    def test_custom_terminal_summary_marks_gamma_unused_and_direct_scatter(self):
        config = generate_mock.MockConfiguration(
            n_qso=3,
            uvx_error_option="custom",
            custom_sigma_ln_dl=0.31,
            sarm_error_option="custom",
            custom_sigma_ln_da=0.22,
            mock_mode="asimov",
            make_diagnostic_plot=False,
        )
        generated = generate_mock.build_catalogue(config)

        summary = self._printed_summary(
            config,
            generated,
            {"csv": Path("/tmp/custom.csv")},
        )

        for text in (
            "UV-X: custom",
            "gamma: not used",
            "sigma_ln_dl: 0.31",
            "SARM: custom",
            "sigma_ln_da: 0.22",
        ):
            self.assertIn(text, summary)
        self.assertNotIn("residual:", summary)
        self.assertNotIn("min/median/max", summary)

    def test_table_terminal_summary_reports_realized_ranges_and_provenance(self):
        config = generate_mock.MockConfiguration(
            n_qso=5,
            z_min=0.5,
            z_max=1.5,
            uvx_error_option="lusso2020_flux_table",
            uvx_delta_dex=0.21,
            flux_measurement_error_correlation=-0.35,
            flux_error_table=str(FLUX_TABLE_FIXTURE),
            sarm_error_option="current",
            mock_mode="asimov",
            make_diagnostic_plot=False,
        )
        generated = generate_mock.build_catalogue(config)
        paths = {
            "csv": Path("/tmp/table.csv"),
            "metadata": Path("/tmp/table_metadata.json"),
        }

        summary = self._printed_summary(config, generated, paths)

        flux = generated.flux_uncertainties
        expected_ranges = (
            (
                "UV uncertainty (dex) min/median/max: "
                f"{np.min(flux.sigma_log_fuv_dex):.6g} / "
                f"{np.median(flux.sigma_log_fuv_dex):.6g} / "
                f"{np.max(flux.sigma_log_fuv_dex):.6g}"
            ),
            (
                "X-ray uncertainty (dex) min/median/max: "
                f"{np.min(flux.sigma_log_fx_dex):.6g} / "
                f"{np.median(flux.sigma_log_fx_dex):.6g} / "
                f"{np.max(flux.sigma_log_fx_dex):.6g}"
            ),
            (
                "sigma_ln_dl min/median/max: "
                f"{np.min(generated.mock.sigma_ln_dl):.6g} / "
                f"{np.median(generated.mock.sigma_ln_dl):.6g} / "
                f"{np.max(generated.mock.sigma_ln_dl):.6g}"
            ),
        )
        for text in (
            "UV-X: lusso2020_flux_table",
            "gamma: 0.6 (used)",
            "delta: 0.21 dex",
            "flux-error correlation: -0.35",
            "SARM: current",
            "sigma_ln_da: 0.3",
            "table support: [0.5, 1.5]",
            f"table: {FLUX_TABLE_FIXTURE.resolve()}",
            "/tmp/table.csv",
            "/tmp/table_metadata.json",
            *expected_ranges,
        ):
            self.assertIn(text, summary)

    def test_table_provenance_is_retained_with_the_built_catalogue(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / FLUX_TABLE_FIXTURE.name
            source_metadata = source.with_name(
                "lusso2020_flux_error_model_metadata.json"
            )
            shutil.copyfile(FLUX_TABLE_FIXTURE, source)
            shutil.copyfile(
                FLUX_TABLE_FIXTURE.with_name(
                    "lusso2020_flux_error_model_metadata.json"
                ),
                source_metadata,
            )
            config = generate_mock.MockConfiguration(
                n_qso=3,
                z_min=0.5,
                z_max=1.5,
                uvx_error_option="lusso2020_flux_table",
                flux_error_table=str(source),
                output_directory=str(Path(directory) / "output"),
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
            generated = generate_mock.build_catalogue(config)
            source.unlink()
            source_metadata.unlink()

            paths = generate_mock.write_catalogue(config, generated)
            uvx = json.loads(paths["metadata"].read_text())["uvx_errors"]

        self.assertEqual(
            uvx["table_sha256"],
            "6a1efa62e4a292f85c38ef9dd28a8c819c3be4eda3e0de60d93d4ce86c7b7104",
        )
        self.assertEqual(uvx["reference"], "test-lusso2020-table-fixture")

    def test_schema_version_2_csv_columns_and_mode_specific_values(self):
        additional = [
            "sigma_log_fuv_dex",
            "sigma_log_fx_dex",
            "sigma_uvx_measurement_dex",
            "sigma_uvx_intrinsic_dex",
            "sigma_uvx_total_dex",
            "corr_ln_uncertainty_sizes",
            "flux_model_effective_count",
            "flux_model_smoothing_width_z",
        ]
        expected_columns = [
            "z",
            "eta_true",
            "dl_true_mpc",
            "dl_observed_mpc",
            "sigma_ln_dl",
            *additional,
            "da_true_mpc",
            "da_observed_mpc",
            "sigma_ln_da",
            "rho",
        ]
        cases = (
            ("current", False),
            ("custom", False),
            ("lusso2020_flux_table", True),
        )
        with tempfile.TemporaryDirectory() as directory:
            for option, table_mode in cases:
                with self.subTest(option=option):
                    config = generate_mock.MockConfiguration(
                        n_qso=3,
                        z_min=0.5,
                        z_max=1.5,
                        uvx_error_option=option,
                        flux_error_table=str(FLUX_TABLE_FIXTURE),
                        output_directory=str(Path(directory) / option),
                        mock_mode="asimov",
                        make_diagnostic_plot=False,
                    )
                    generated = generate_mock.build_catalogue(config)
                    paths = generate_mock.write_catalogue(config, generated)
                    with paths["csv"].open(newline="") as handle:
                        reader = csv.DictReader(handle)
                        rows = list(reader)

                    self.assertEqual(reader.fieldnames, expected_columns)
                    for row in rows:
                        if table_mode:
                            self.assertTrue(all(row[name] != "" for name in additional))
                        elif option == "current":
                            self.assertEqual(
                                float(row["sigma_uvx_total_dex"]),
                                generated.uvx_errors.sigma_uvx_dex,
                            )
                            self.assertTrue(
                                all(
                                    row[name] == ""
                                    for name in additional
                                    if name != "sigma_uvx_total_dex"
                                )
                            )
                        else:
                            self.assertTrue(
                                all(row[name] == "" for name in additional)
                            )

    def test_schema_version_2_metadata_separates_errors_and_records_table_provenance(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config = generate_mock.MockConfiguration(
                n_qso=3,
                z_min=0.5,
                z_max=1.5,
                uvx_error_option="lusso2020_flux_table",
                flux_error_table=str(FLUX_TABLE_FIXTURE),
                sarm_error_option="future",
                output_directory=directory,
                mock_mode="random",
                make_diagnostic_plot=False,
            )
            generated = generate_mock.build_catalogue(config)
            paths = generate_mock.write_catalogue(config, generated)
            metadata = json.loads(paths["metadata"].read_text())
            uvx = metadata["uvx_errors"]

        self.assertEqual(metadata["schema_version"], 2)
        self.assertNotIn("resolved_errors", metadata)
        self.assertEqual(
            set(uvx),
            {
                "option",
                "gamma",
                "gamma_used",
                "sigma_uvx_dex",
                "custom_sigma_ln_dl",
                "delta_dex",
                "flux_measurement_error_correlation",
                "distribution",
                "interpolation",
                "table_path",
                "table_sha256",
                "table_schema_version",
                "reference",
                "supported_redshift_interval",
                "redshifts_generated_independently",
            },
        )
        self.assertEqual(uvx["option"], "lusso2020_flux_table")
        self.assertEqual(uvx["gamma"], 0.60)
        self.assertTrue(uvx["gamma_used"])
        self.assertIsNone(uvx["sigma_uvx_dex"])
        self.assertIsNone(uvx["custom_sigma_ln_dl"])
        self.assertEqual(uvx["delta_dex"], 0.21)
        self.assertEqual(uvx["flux_measurement_error_correlation"], 0.0)
        self.assertEqual(
            uvx["distribution"],
            "correlated lognormal per-object flux uncertainties",
        )
        self.assertEqual(
            uvx["interpolation"],
            "log-linear medians and linear distribution parameters in redshift",
        )
        self.assertEqual(uvx["table_path"], str(FLUX_TABLE_FIXTURE.resolve()))
        self.assertEqual(
            uvx["table_sha256"],
            "6a1efa62e4a292f85c38ef9dd28a8c819c3be4eda3e0de60d93d4ce86c7b7104",
        )
        self.assertEqual(uvx["table_schema_version"], 1)
        self.assertEqual(uvx["reference"], "test-lusso2020-table-fixture")
        self.assertEqual(uvx["supported_redshift_interval"], [0.5, 1.5])
        self.assertTrue(uvx["redshifts_generated_independently"])
        self.assertEqual(
            metadata["sarm_errors"],
            {"option": "future", "sigma_ln_da": 0.2},
        )
        self.assertEqual(metadata["distance_unit"], "Mpc")
        self.assertEqual(
            metadata["error_unit"],
            "natural-log distance standard deviation",
        )
        self.assertTrue(metadata["errors_include_intrinsic"])
        self.assertEqual(
            metadata["mock_generation"]["seed_semantics"],
            "independent redshift, distance-noise, and flux-uncertainty "
            "streams derived from master_seed",
        )
        self.assertEqual(
            metadata["ddr"]["definition"],
            "eta(z) = DL(z) / ((1 + z)^2 DA(z))",
        )
        self.assertIn("model_equation", metadata["ddr"])
        self.assertEqual(metadata["cosmology"]["model"], "flat_lcdm")
        self.assertIn("sigma_uvx_total_dex", metadata["columns"])

    def test_inactive_uvx_metadata_values_are_null_for_named_and_custom_modes(self):
        cases = (
            ("current", 0.24, True),
            ("custom", None, False),
        )
        inactive_table_keys = (
            "delta_dex",
            "flux_measurement_error_correlation",
            "distribution",
            "interpolation",
            "table_path",
            "table_sha256",
            "table_schema_version",
            "reference",
            "supported_redshift_interval",
            "redshifts_generated_independently",
        )
        with tempfile.TemporaryDirectory() as directory:
            for option, total_dex, gamma_used in cases:
                with self.subTest(option=option):
                    config = generate_mock.MockConfiguration(
                        n_qso=2,
                        uvx_error_option=option,
                        output_directory=str(Path(directory) / option),
                        mock_mode="asimov",
                        make_diagnostic_plot=False,
                    )
                    generated = generate_mock.build_catalogue(config)
                    paths = generate_mock.write_catalogue(config, generated)
                    uvx = json.loads(paths["metadata"].read_text())["uvx_errors"]

                    self.assertEqual(uvx["option"], option)
                    self.assertEqual(uvx["gamma"], 0.60)
                    self.assertEqual(uvx["gamma_used"], gamma_used)
                    self.assertEqual(
                        uvx["sigma_uvx_dex"],
                        total_dex,
                    )
                    self.assertEqual(
                        uvx["custom_sigma_ln_dl"],
                        0.30 if option == "custom" else None,
                    )
                    for name in inactive_table_keys:
                        self.assertIsNone(uvx[name])

    def test_legacy_error_overflow_fails_before_any_output_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "catalogue"
            config = generate_mock.MockConfiguration(
                n_qso=3,
                uvx_error_option="custom",
                sarm_error_option="custom",
                custom_sigma_ln_dl=1000.0,
                custom_sigma_ln_da=1000.0,
                output_directory=str(output_directory),
                mock_mode="asimov",
                make_diagnostic_plot=False,
                overwrite=True,
            )
            generated = generate_mock.build_catalogue(config)
            output_directory.mkdir()
            csv_path = output_directory / f"{generate_mock.dataset_stem(config)}.csv"
            csv_path.write_bytes(b"existing catalogue")

            error = None
            try:
                generate_mock.write_catalogue(config, generated)
            except Exception as exc:
                error = exc

            self.assertEqual(csv_path.read_bytes(), b"existing catalogue")
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {csv_path.name},
            )
            self.assertIsInstance(error, ValueError)
            self.assertRegex(str(error), "legacy absolute.*finite")

    def test_metadata_serialization_failure_happens_before_any_output_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "catalogue"
            config = generate_mock.MockConfiguration(
                n_qso=3,
                output_directory=str(output_directory),
                mock_mode="asimov",
                make_diagnostic_plot=False,
                overwrite=True,
            )
            generated = generate_mock.build_catalogue(config)
            invalid_generated = replace(
                generated,
                uvx_errors=replace(
                    generated.uvx_errors,
                    sigma_uvx_dex=np.nan,
                ),
            )
            output_directory.mkdir()
            csv_path = output_directory / f"{generate_mock.dataset_stem(config)}.csv"
            csv_path.write_bytes(b"existing catalogue")

            with self.assertRaises(ValueError):
                generate_mock.write_catalogue(config, invalid_generated)

            self.assertEqual(csv_path.read_bytes(), b"existing catalogue")
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {csv_path.name},
            )

    def test_metadata_records_effective_cosmology_parameters_for_each_model(self):
        cases = (
            ("flat_lcdm", 0.0, -1.0),
            ("lcdm", 0.12, -1.0),
            ("flat_wcdm", 0.0, -0.8),
        )
        with tempfile.TemporaryDirectory() as directory:
            for model, expected_omega_k, expected_w0 in cases:
                with self.subTest(model=model):
                    config = generate_mock.MockConfiguration(
                        n_qso=3,
                        cosmology_model=model,
                        omega_k=0.12,
                        w0=-0.8,
                        output_directory=str(Path(directory) / model),
                        mock_mode="asimov",
                        make_diagnostic_plot=False,
                    )
                    generated = generate_mock.build_catalogue(config)
                    paths = generate_mock.write_catalogue(config, generated)
                    metadata = json.loads(paths["metadata"].read_text())

                    self.assertEqual(
                        metadata["cosmology"]["omega_k"], expected_omega_k
                    )
                    self.assertEqual(metadata["cosmology"]["w0"], expected_w0)
                    self.assertEqual(metadata["configuration"]["omega_k"], 0.12)
                    self.assertEqual(metadata["configuration"]["w0"], -0.8)

    def test_diagnostic_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            off = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=str(Path(directory) / "off"),
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
            off_paths = generate_mock.write_catalogue(
                off, generate_mock.build_catalogue(off)
            )
            self.assertNotIn("diagnostic", off_paths)

            on = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=str(Path(directory) / "on"),
                mock_mode="asimov",
                make_diagnostic_plot=True,
            )
            on_paths = generate_mock.write_catalogue(
                on, generate_mock.build_catalogue(on)
            )
            self.assertIn("diagnostic", on_paths)
            self.assertTrue(on_paths["diagnostic"].exists())

    def test_diagnostic_receives_authoritative_metadata_and_dataset_name(self):
        with tempfile.TemporaryDirectory() as directory:
            config = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=directory,
                mock_mode="asimov",
                make_diagnostic_plot=True,
            )
            generated = generate_mock.build_catalogue(config)
            with mock.patch(
                "generate_mock.save_paired_mock_diagnostic"
            ) as diagnostic:
                diagnostic.side_effect = (
                    lambda path, mock, **kwargs: Path(path).write_bytes(b"plot")
                )
                generate_mock.write_catalogue(config, generated)

            _, positional_mock = diagnostic.call_args.args
            keyword = diagnostic.call_args.kwargs
            self.assertIs(positional_mock, generated.mock)
            self.assertEqual(
                keyword["title"],
                generate_mock.dataset_stem(config),
            )
            self.assertEqual(
                keyword["metadata"]["configuration"]["n_qso"],
                4,
            )
            self.assertEqual(keyword["metadata"]["uvx_errors"]["option"], "current")
            self.assertEqual(
                keyword["metadata"].get("diagnostic_uncertainty_summary"),
                {
                    "sigma_ln_dl": {
                        "min": generated.uvx_errors.sigma_ln_dl,
                        "median": generated.uvx_errors.sigma_ln_dl,
                        "max": generated.uvx_errors.sigma_ln_dl,
                    }
                },
            )
            self.assertEqual(
                keyword["metadata"]["cosmology"]["model"],
                "flat_lcdm",
            )
            self.assertEqual(keyword["metadata"]["ddr"]["model"], "M1")

    def test_table_diagnostic_receives_precomputed_sigma_ln_dl_range(self):
        with tempfile.TemporaryDirectory() as directory:
            config = generate_mock.MockConfiguration(
                n_qso=5,
                z_min=0.5,
                z_max=1.5,
                uvx_error_option="lusso2020_flux_table",
                flux_error_table=str(FLUX_TABLE_FIXTURE),
                output_directory=directory,
                mock_mode="asimov",
                make_diagnostic_plot=True,
            )
            generated = generate_mock.build_catalogue(config)
            with mock.patch(
                "generate_mock.save_paired_mock_diagnostic"
            ) as diagnostic:
                diagnostic.side_effect = (
                    lambda path, mock, **kwargs: Path(path).write_bytes(b"plot")
                )
                generate_mock.write_catalogue(config, generated)

            diagnostic_summary = diagnostic.call_args.kwargs["metadata"].get(
                "diagnostic_uncertainty_summary"
            )

        self.assertIsNotNone(diagnostic_summary)
        summary = diagnostic_summary["sigma_ln_dl"]
        self.assertEqual(
            summary["min"],
            float(np.min(generated.mock.sigma_ln_dl)),
        )
        self.assertEqual(
            summary["median"],
            float(np.median(generated.mock.sigma_ln_dl)),
        )
        self.assertEqual(summary["max"], float(np.max(generated.mock.sigma_ln_dl)))

    def test_all_primary_outputs_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            config = generate_mock.MockConfiguration(
                n_qso=6,
                output_directory=directory,
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
            generated = generate_mock.build_catalogue(config)
            paths = generate_mock.write_catalogue(config, generated)

            self.assertEqual(
                set(paths), {"csv", "metadata", "npz", "dl_text", "da_text"}
            )
            table = np.genfromtxt(paths["csv"], delimiter=",", names=True)
            paired = load_paired_log_mock(paths["npz"])
            legacy_dl = np.genfromtxt(paths["dl_text"], names=True)
            legacy_da = np.genfromtxt(paths["da_text"], names=True)
            metadata = json.loads(paths["metadata"].read_text())

            np.testing.assert_allclose(table["z"], paired.redshift)
            np.testing.assert_allclose(
                table["dl_observed_mpc"], paired.dl_observed
            )
            np.testing.assert_allclose(
                table["da_observed_mpc"], paired.da_observed
            )
            self.assertEqual(
                legacy_dl.dtype.names, ("z", "dl_mpc", "sigma_dl_mpc")
            )
            self.assertEqual(
                legacy_da.dtype.names, ("z", "da_mpc", "sigma_da_mpc")
            )
            np.testing.assert_allclose(legacy_dl["z"], table["z"])
            np.testing.assert_allclose(legacy_dl["dl_mpc"], paired.dl_observed)
            np.testing.assert_allclose(
                legacy_dl["sigma_dl_mpc"],
                paired.dl_true * np.sqrt(np.expm1(paired.sigma_ln_dl**2)),
            )
            np.testing.assert_allclose(legacy_da["z"], table["z"])
            np.testing.assert_allclose(legacy_da["da_mpc"], paired.da_observed)
            np.testing.assert_allclose(
                legacy_da["sigma_da_mpc"],
                paired.da_true * np.sqrt(np.expm1(paired.sigma_ln_da**2)),
            )
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["distance_unit"], "Mpc")
            self.assertTrue(metadata["errors_include_intrinsic"])
            self.assertEqual(
                metadata["ddr"]["definition"],
                "eta(z) = DL(z) / ((1 + z)^2 DA(z))",
            )
            self.assertIn("seed_semantics", metadata["mock_generation"])
            self.assertEqual(
                metadata["source_files"]["theory"],
                "tools/theory.py",
            )

    def test_overwrite_false_stops_before_replacing_any_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=directory,
                mock_mode="asimov",
                make_diagnostic_plot=False,
            )
            generated = generate_mock.build_catalogue(config)
            paths = generate_mock.write_catalogue(config, generated)
            original = paths["csv"].read_bytes()
            with self.assertRaisesRegex(FileExistsError, "OVERWRITE"):
                generate_mock.write_catalogue(config, generated)
            self.assertEqual(paths["csv"].read_bytes(), original)

    def test_no_overwrite_promotion_preserves_a_late_concurrent_target(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "catalogue"
            config = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=str(output_directory),
                mock_mode="asimov",
                make_diagnostic_plot=True,
                overwrite=False,
            )
            generated = generate_mock.build_catalogue(config)
            late_target = generate_mock._target_paths(config)["csv"]
            concurrent_bytes = b"created by concurrent writer"
            real_link = os.link
            injected = False

            def inject_target_before_link(source, destination):
                nonlocal injected
                if not injected:
                    Path(destination).write_bytes(concurrent_bytes)
                    injected = True
                return real_link(source, destination)

            with mock.patch("os.link", side_effect=inject_target_before_link):
                with self.assertRaisesRegex(
                    FileExistsError,
                    "appeared during staging",
                ):
                    generate_mock.write_catalogue(config, generated)

            self.assertTrue(injected)
            self.assertEqual(late_target.read_bytes(), concurrent_bytes)
            self.assertEqual(
                set(output_directory.iterdir()),
                {late_target},
            )

    def test_no_overwrite_rollback_preserves_replacement_of_an_earlier_link(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "catalogue"
            config = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=str(output_directory),
                mock_mode="asimov",
                make_diagnostic_plot=True,
                overwrite=False,
            )
            generated = generate_mock.build_catalogue(config)
            final_paths = list(generate_mock._target_paths(config).values())
            replaced_target, later_conflict = final_paths[:2]
            concurrent_bytes = b"unrelated replacement after first link"
            conflict_bytes = b"unrelated later conflict"
            real_link = os.link
            link_calls = 0
            linked_identity = None
            replacement_identity = None

            def replace_first_link_then_conflict(source, destination):
                nonlocal link_calls, linked_identity, replacement_identity
                link_calls += 1
                destination = Path(destination)
                if link_calls == 1:
                    result = real_link(source, destination)
                    stat_result = destination.lstat()
                    linked_identity = (stat_result.st_dev, stat_result.st_ino)
                    replacement = destination.with_name(".concurrent-replacement")
                    replacement.write_bytes(concurrent_bytes)
                    os.replace(replacement, destination)
                    stat_result = destination.lstat()
                    replacement_identity = (
                        stat_result.st_dev,
                        stat_result.st_ino,
                    )
                    return result
                if link_calls == 2:
                    destination.write_bytes(conflict_bytes)
                return real_link(source, destination)

            with mock.patch(
                "os.link",
                side_effect=replace_first_link_then_conflict,
            ):
                with self.assertRaisesRegex(
                    FileExistsError,
                    "appeared during staging",
                ):
                    generate_mock.write_catalogue(config, generated)

            self.assertNotEqual(linked_identity, replacement_identity)
            self.assertTrue(
                replaced_target.exists(),
                "rollback deleted another writer's replacement",
            )
            self.assertEqual(replaced_target.read_bytes(), concurrent_bytes)
            current_stat = replaced_target.lstat()
            self.assertEqual(
                (current_stat.st_dev, current_stat.st_ino),
                replacement_identity,
            )
            self.assertEqual(later_conflict.read_bytes(), conflict_bytes)
            self.assertEqual(
                set(output_directory.iterdir()),
                {replaced_target, later_conflict},
            )

    def test_diagnostic_only_conflict_stops_before_primary_output_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "catalogue"
            config = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=str(output_directory),
                mock_mode="asimov",
                make_diagnostic_plot=True,
                overwrite=False,
            )
            diagnostic = (
                output_directory
                / f"{generate_mock.dataset_stem(config)}_diagnostic.png"
            )
            output_directory.mkdir()
            diagnostic.write_bytes(b"existing diagnostic")

            with self.assertRaisesRegex(FileExistsError, str(diagnostic)):
                generate_mock.write_catalogue(
                    config, generate_mock.build_catalogue(config)
                )

            self.assertEqual(diagnostic.read_bytes(), b"existing diagnostic")
            self.assertEqual(
                {path.name for path in output_directory.iterdir()}, {diagnostic.name}
            )

    def test_disabled_diagnostic_is_still_an_output_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "catalogue"
            config = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=str(output_directory),
                mock_mode="asimov",
                make_diagnostic_plot=False,
                overwrite=False,
            )
            diagnostic = (
                output_directory
                / f"{generate_mock.dataset_stem(config)}_diagnostic.png"
            )
            output_directory.mkdir()
            diagnostic.write_bytes(b"existing diagnostic")

            with self.assertRaisesRegex(FileExistsError, str(diagnostic)):
                generate_mock.write_catalogue(
                    config, generate_mock.build_catalogue(config)
                )

            self.assertEqual(diagnostic.read_bytes(), b"existing diagnostic")
            self.assertEqual(
                {path.name for path in output_directory.iterdir()}, {diagnostic.name}
            )

    def test_overwrite_with_diagnostic_disabled_removes_same_stem_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            enabled = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=directory,
                mock_mode="asimov",
                make_diagnostic_plot=True,
            )
            enabled_paths = generate_mock.write_catalogue(
                enabled, generate_mock.build_catalogue(enabled)
            )
            diagnostic = enabled_paths["diagnostic"]
            self.assertTrue(diagnostic.exists())

            disabled = replace(
                enabled,
                make_diagnostic_plot=False,
                overwrite=True,
            )
            disabled_paths = generate_mock.write_catalogue(
                disabled, generate_mock.build_catalogue(disabled)
            )

            self.assertNotIn("diagnostic", disabled_paths)
            self.assertFalse(diagnostic.exists())
            self.assertTrue(all(path.exists() for path in disabled_paths.values()))

    def test_new_output_failure_during_diagnostic_staging_leaves_no_final_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "catalogue"
            config = generate_mock.MockConfiguration(
                n_qso=3,
                output_directory=str(output_directory),
                mock_mode="asimov",
                make_diagnostic_plot=True,
            )
            with mock.patch(
                "generate_mock.save_paired_mock_diagnostic",
                side_effect=OSError("disk quota"),
            ):
                with self.assertRaisesRegex(OSError, "disk quota"):
                    generate_mock.write_catalogue(
                        config,
                        generate_mock.build_catalogue(config),
                    )

            self.assertTrue(output_directory.exists())
            self.assertEqual(list(output_directory.iterdir()), [])

    def test_overwrite_failure_during_diagnostic_staging_preserves_all_old_outputs(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "catalogue"
            config = generate_mock.MockConfiguration(
                n_qso=3,
                output_directory=str(output_directory),
                mock_mode="asimov",
                make_diagnostic_plot=True,
                overwrite=True,
            )
            output_directory.mkdir()
            final_paths = generate_mock._target_paths(config)
            expected = {}
            for index, path in enumerate(final_paths.values()):
                content = f"old-{index}".encode()
                path.write_bytes(content)
                expected[path] = content

            with mock.patch(
                "generate_mock.save_paired_mock_diagnostic",
                side_effect=OSError("disk quota"),
            ):
                with self.assertRaisesRegex(OSError, "disk quota"):
                    generate_mock.write_catalogue(
                        config,
                        generate_mock.build_catalogue(config),
                    )

            self.assertEqual(
                {path: path.read_bytes() for path in expected},
                expected,
            )
            self.assertEqual(
                set(output_directory.iterdir()),
                set(expected),
            )

    def test_successful_overwrite_promotes_the_complete_same_stem_set(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=directory,
                random_seed=11,
                make_diagnostic_plot=False,
            )
            old_paths = generate_mock.write_catalogue(
                initial,
                generate_mock.build_catalogue(initial),
            )
            old_csv = old_paths["csv"].read_bytes()
            replacement = replace(
                initial,
                random_seed=12,
                overwrite=True,
            )
            new_paths = generate_mock.write_catalogue(
                replacement,
                generate_mock.build_catalogue(replacement),
            )

            self.assertEqual(set(new_paths), set(old_paths))
            self.assertNotEqual(new_paths["csv"].read_bytes(), old_csv)
            self.assertTrue(all(path.exists() for path in new_paths.values()))
            self.assertEqual(
                set(Path(directory).iterdir()),
                set(new_paths.values()),
            )

    def test_keyboard_interrupt_during_each_promotion_restores_all_outputs(self):
        for interrupted_promotion in range(1, 7):
            with self.subTest(interrupted_promotion=interrupted_promotion):
                with tempfile.TemporaryDirectory() as directory:
                    output_directory = Path(directory) / "catalogue"
                    config = generate_mock.MockConfiguration(
                        n_qso=3,
                        output_directory=str(output_directory),
                        mock_mode="asimov",
                        make_diagnostic_plot=True,
                        overwrite=True,
                    )
                    output_directory.mkdir()
                    final_paths = generate_mock._target_paths(config)
                    expected = {}
                    for index, path in enumerate(final_paths.values()):
                        content = f"original-{index}".encode()
                        path.write_bytes(content)
                        expected[path] = content

                    original_replace = Path.replace
                    promotion_count = 0

                    def interrupt_one_promotion(source, target):
                        nonlocal promotion_count
                        source = Path(source)
                        target = Path(target)
                        if (
                            source.parent.name.startswith(
                                f".{generate_mock.dataset_stem(config)}.staging-"
                            )
                            and source.parent.name != "previous"
                            and target in expected
                        ):
                            promotion_count += 1
                            if promotion_count == interrupted_promotion:
                                raise KeyboardInterrupt(
                                    f"interrupted promotion {interrupted_promotion}"
                                )
                        return original_replace(source, target)

                    with mock.patch.object(
                        Path,
                        "replace",
                        autospec=True,
                        side_effect=interrupt_one_promotion,
                    ):
                        with self.assertRaisesRegex(
                            KeyboardInterrupt,
                            f"interrupted promotion {interrupted_promotion}",
                        ):
                            generate_mock.write_catalogue(
                                config,
                                generate_mock.build_catalogue(config),
                            )

                    actual = {
                        path: path.read_bytes() if path.exists() else None
                        for path in expected
                    }
                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        set(output_directory.iterdir()),
                        set(expected),
                    )

    def test_failed_disabled_diagnostic_overwrite_keeps_old_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            enabled = generate_mock.MockConfiguration(
                n_qso=4,
                output_directory=directory,
                mock_mode="asimov",
                make_diagnostic_plot=True,
            )
            with mock.patch(
                "generate_mock.save_paired_mock_diagnostic",
                side_effect=lambda path, *args, **kwargs: Path(path).write_bytes(
                    b"old diagnostic"
                ),
            ):
                enabled_paths = generate_mock.write_catalogue(
                    enabled,
                    generate_mock.build_catalogue(enabled),
                )
            diagnostic = enabled_paths["diagnostic"]
            old_diagnostic = diagnostic.read_bytes()
            disabled = replace(
                enabled,
                make_diagnostic_plot=False,
                overwrite=True,
            )

            with mock.patch(
                "generate_mock.np.savetxt",
                side_effect=OSError("disk quota"),
            ):
                with self.assertRaisesRegex(OSError, "disk quota"):
                    generate_mock.write_catalogue(
                        disabled,
                        generate_mock.build_catalogue(disabled),
                    )

            self.assertEqual(diagnostic.read_bytes(), old_diagnostic)

    def test_failed_rollback_retains_backups_and_restores_every_other_original(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "catalogue"
            config = generate_mock.MockConfiguration(
                n_qso=3,
                output_directory=str(output_directory),
                mock_mode="asimov",
                make_diagnostic_plot=False,
                overwrite=True,
            )
            output_directory.mkdir()
            final_paths = generate_mock._target_paths(config)
            expected = {}
            for index, path in enumerate(final_paths.values()):
                content = f"old-{index}".encode()
                path.write_bytes(content)
                expected[path] = content

            original_replace = Path.replace
            failures = {"promotion": False, "restoration": False}

            def injected_replace(source, target):
                source = Path(source)
                target = Path(target)
                if (
                    not failures["promotion"]
                    and source.parent.name.startswith(
                        f".{generate_mock.dataset_stem(config)}.staging-"
                    )
                    and source.name.endswith("_paired_dataset.npz")
                ):
                    failures["promotion"] = True
                    raise OSError("promotion failed")
                if (
                    failures["promotion"]
                    and not failures["restoration"]
                    and source.parent.name == "previous"
                    and target == final_paths["metadata"]
                ):
                    failures["restoration"] = True
                    raise OSError("restoration failed")
                return original_replace(source, target)

            with mock.patch.object(Path, "replace", autospec=True) as replace_path:
                replace_path.side_effect = injected_replace
                with self.assertRaisesRegex(
                    RuntimeError,
                    "recovery files retained at",
                ):
                    generate_mock.write_catalogue(
                        config,
                        generate_mock.build_catalogue(config),
                    )

            recovery_directories = [
                path
                for path in output_directory.iterdir()
                if path.is_dir() and ".staging-" in path.name
            ]
            self.assertEqual(len(recovery_directories), 1)
            backup_metadata = (
                recovery_directories[0]
                / "previous"
                / final_paths["metadata"].name
            )
            self.assertEqual(
                backup_metadata.read_bytes(),
                expected[final_paths["metadata"]],
            )
            for name, path in final_paths.items():
                if name != "metadata":
                    self.assertEqual(path.read_bytes(), expected[path])
