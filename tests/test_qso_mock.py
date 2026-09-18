import unittest
import warnings
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import interp1d

from tools.ddr_statistics import observed_log_eta
from tools.QSO_specifications import QSOSpecs
from tools.qso_mock import (
    QSOMockGenerator,
    generate_paired_log_mock,
    load_paired_log_mock,
    save_paired_log_mock,
)


class PairedLogMockTests(unittest.TestCase):
    def setUp(self):
        self.z = np.array([0.1, 0.5, 1.0, 2.0])
        self.da = np.array([100.0, 500.0, 900.0, 1100.0])

    def test_asimov_mock_recovers_injected_eta(self):
        mock = generate_paired_log_mock(
            self.z,
            self.da,
            eta=0.925,
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            mode="asimov",
        )
        np.testing.assert_allclose(mock.da_observed, self.da)
        np.testing.assert_allclose(mock.dl_observed, mock.dl_true)
        np.testing.assert_allclose(
            observed_log_eta(mock.dl_observed, mock.da_observed, self.z),
            np.log(0.925),
        )

    def test_random_mock_is_seeded_and_positive(self):
        kwargs = dict(
            redshift=self.z,
            da_true=self.da,
            eta=1.0,
            sigma_ln_dl=0.6,
            sigma_ln_da=0.3,
            mode="random",
            seed=27,
        )
        first = generate_paired_log_mock(**kwargs)
        second = generate_paired_log_mock(**kwargs)
        np.testing.assert_array_equal(first.dl_observed, second.dl_observed)
        np.testing.assert_array_equal(first.da_observed, second.da_observed)
        self.assertTrue(np.all(first.dl_observed > 0))
        self.assertTrue(np.all(first.da_observed > 0))

    def test_random_mock_rejects_underflowed_observed_distance(self):
        with self.assertRaisesRegex(ValueError, "finite.*strictly positive"):
            generate_paired_log_mock(
                self.z[:1],
                self.da[:1],
                eta=1.0,
                sigma_ln_dl=2000.0,
                sigma_ln_da=0.0,
                mode="random",
                seed=4,
            )

    def test_random_mock_rejects_overflowed_observed_distance(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "finite.*strictly positive"):
                generate_paired_log_mock(
                    self.z[:1],
                    self.da[:1],
                    eta=1.0,
                    sigma_ln_dl=0.0,
                    sigma_ln_da=3000.0,
                    rho=1.0,
                    mode="random",
                    seed=1,
                )

    def test_caller_supplied_rng_controls_draw_but_master_seed_is_recorded(self):
        kwargs = dict(
            redshift=self.z,
            da_true=self.da,
            eta=0.925,
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            mode="random",
            seed=27,
        )
        noise_rng = np.random.default_rng(np.random.SeedSequence(27).spawn(2)[1])
        common = noise_rng.standard_normal(self.z.size)
        independent = noise_rng.standard_normal(self.z.size)
        dl_true = 0.925 * (1.0 + self.z) ** 2 * self.da
        expected_dl = dl_true * np.exp(0.3 * common)
        expected_da = self.da * np.exp(0.2 * independent)
        first = generate_paired_log_mock(
            **kwargs, rng=np.random.default_rng(np.random.SeedSequence(27).spawn(2)[1])
        )
        second = generate_paired_log_mock(
            **kwargs, rng=np.random.default_rng(np.random.SeedSequence(27).spawn(2)[1])
        )
        np.testing.assert_array_equal(first.dl_observed, second.dl_observed)
        np.testing.assert_array_equal(first.dl_observed, expected_dl)
        np.testing.assert_array_equal(first.da_observed, expected_da)
        self.assertEqual(first.seed, 27)

    def test_rng_must_be_a_numpy_generator(self):
        with self.assertRaisesRegex(TypeError, "Generator"):
            generate_paired_log_mock(
                self.z,
                self.da,
                eta=1.0,
                sigma_ln_dl=0.3,
                sigma_ln_da=0.2,
                mode="random",
                rng=object(),
            )

    def test_random_mock_supports_same_object_correlation(self):
        z = np.linspace(0.1, 2.0, 50_000)
        da = np.full_like(z, 1000.0)
        mock = generate_paired_log_mock(
            z,
            da,
            eta=1.0,
            sigma_ln_dl=0.4,
            sigma_ln_da=0.2,
            rho=0.6,
            mode="random",
            seed=4,
        )
        dl_residual = np.log(mock.dl_observed / mock.dl_true)
        da_residual = np.log(mock.da_observed / mock.da_true)
        self.assertAlmostEqual(np.corrcoef(dl_residual, da_residual)[0, 1], 0.6, places=2)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            generate_paired_log_mock(
                self.z, self.da, eta=1.0, sigma_ln_dl=0.1,
                sigma_ln_da=0.1, mode="validated"
            )

    def test_nonfinite_mock_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            generate_paired_log_mock(
                self.z, self.da, eta=np.nan, sigma_ln_dl=0.1,
                sigma_ln_da=0.1
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            generate_paired_log_mock(
                self.z, self.da, eta=1.0, sigma_ln_dl=np.inf,
                sigma_ln_da=0.1
            )

    def test_npz_round_trip_preserves_error_semantics(self):
        import tempfile
        from pathlib import Path

        mock = generate_paired_log_mock(
            self.z, self.da, eta=0.925, sigma_ln_dl=0.3,
            sigma_ln_da=0.2, rho=0.1, mode="random", seed=8
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.npz"
            save_paired_log_mock(path, mock)
            loaded = load_paired_log_mock(path)
        np.testing.assert_array_equal(loaded.redshift, mock.redshift)
        np.testing.assert_array_equal(loaded.dl_observed, mock.dl_observed)
        np.testing.assert_array_equal(loaded.sigma_ln_dl, mock.sigma_ln_dl)
        self.assertEqual(loaded.mode, "random")
        self.assertEqual(loaded.seed, 8)
        self.assertTrue(loaded.errors_include_intrinsic)

    def test_old_minimal_npz_remains_loadable(self):
        import tempfile
        from pathlib import Path

        mock = generate_paired_log_mock(
            self.z,
            self.da,
            eta=0.925,
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            rho=0.1,
            mode="random",
            seed=8,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_minimal.npz"
            np.savez_compressed(
                path,
                redshift=mock.redshift,
                dl_true=mock.dl_true,
                da_true=mock.da_true,
                dl_observed=mock.dl_observed,
                da_observed=mock.da_observed,
                sigma_ln_dl=mock.sigma_ln_dl,
                sigma_ln_da=mock.sigma_ln_da,
                rho=mock.rho,
                mode=np.asarray(mock.mode),
                seed=np.asarray(mock.seed, dtype=int),
            )
            loaded = load_paired_log_mock(path)

        np.testing.assert_array_equal(loaded.redshift, mock.redshift)
        np.testing.assert_array_equal(loaded.dl_observed, mock.dl_observed)
        self.assertTrue(loaded.errors_include_intrinsic)

    def test_extended_npz_stores_flux_components_without_changing_load_contract(self):
        import tempfile
        from pathlib import Path

        mock = generate_paired_log_mock(
            self.z,
            self.da,
            eta=0.925,
            sigma_ln_dl=np.array([0.30, 0.31, 0.32, 0.33]),
            sigma_ln_da=0.2,
            mode="asimov",
        )
        flux_arrays = {
            "sigma_log_fuv_dex": np.array([0.10, 0.11, 0.12, 0.13]),
            "sigma_log_fx_dex": np.array([0.20, 0.21, 0.22, 0.23]),
            "sigma_uvx_measurement_dex": np.array([0.22, 0.23, 0.24, 0.25]),
            "sigma_uvx_intrinsic_dex": np.array([0.21, 0.21, 0.21, 0.21]),
            "sigma_uvx_total_dex": np.array([0.30, 0.31, 0.32, 0.33]),
            "corr_ln_uncertainty_sizes": np.array([0.1, 0.2, 0.3, 0.4]),
            "flux_model_effective_count": np.array([90.0, 80.0, 70.0, 60.0]),
            "flux_model_smoothing_width_z": np.array([0.2, 0.2, 0.3, 0.3]),
        }
        realization = SimpleNamespace(
            **{
                **{
                    name: values
                    for name, values in flux_arrays.items()
                    if not name.startswith("flux_model_")
                },
                "effective_reference_count": flux_arrays[
                    "flux_model_effective_count"
                ],
                "smoothing_width_z": flux_arrays[
                    "flux_model_smoothing_width_z"
                ],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extended.npz"
            save_paired_log_mock(
                path,
                mock,
                flux_uncertainties=realization,
            )
            with np.load(path, allow_pickle=False) as data:
                for name, expected in flux_arrays.items():
                    self.assertIn(name, data.files)
                    np.testing.assert_array_equal(data[name], expected)
            loaded = load_paired_log_mock(path)

        self.assertEqual(type(loaded).__name__, "PairedQSOMock")
        np.testing.assert_array_equal(loaded.redshift, mock.redshift)
        np.testing.assert_array_equal(loaded.sigma_ln_dl, mock.sigma_ln_dl)

    def test_legacy_generator_writes_new_paired_dataset(self):
        import tempfile
        from pathlib import Path

        c = 299792.458
        h0 = 70.0
        z_grid = np.linspace(0.0, 2.0, 50)

        def da(z):
            return 1000.0 / (1.0 + np.asarray(z))

        def dimensionless_dl(z):
            z = np.asarray(z)
            return h0 / c * 0.925 * (1.0 + z) ** 2 * da(z)

        with tempfile.TemporaryDirectory() as directory:
            outpath = str(Path(directory) / "mock")
            generator = QSOMockGenerator(
                {"H0": h0},
                {"redshifts": [0.2, 0.8, 1.5]},
                z_grid,
                outpath,
                dl_func=dimensionless_dl,
                da_func=da,
                error_scenario="targeted",
                mode="asimov",
                seed=3,
            )
            self.assertTrue(Path(outpath + "_paired_dataset.npz").exists())
            paired = load_paired_log_mock(outpath + "_paired_dataset.npz")
        np.testing.assert_allclose(
            observed_log_eta(paired.dl_observed, paired.da_observed, paired.redshift),
            np.log(0.925),
        )
        self.assertAlmostEqual(generator.mock()["sigma_ln_da"][0], 0.15)

    def test_generator_uses_independent_child_streams_for_redshift_and_noise(self):
        import tempfile
        from pathlib import Path

        c = 299792.458
        h0 = 70.0
        eta = 0.925
        seed = 27
        specs = {"z_min": 0.2, "z_max": 1.8, "n_QSO": 4}
        z_grid = np.linspace(0.0, 2.0, 50)

        def da(z):
            return 1000.0 / (1.0 + np.asarray(z))

        def dimensionless_dl(z):
            z = np.asarray(z)
            return h0 / c * eta * (1.0 + z) ** 2 * da(z)

        redshift_stream, noise_stream = np.random.SeedSequence(seed).spawn(2)
        expected_redshift = QSOSpecs.redshift_distribution(
            specs, rng=np.random.default_rng(redshift_stream), mode="random"
        )
        expected_dl_true = interp1d(
            z_grid, c / h0 * dimensionless_dl(z_grid)
        )(expected_redshift)
        expected_eta = expected_dl_true / (
            (1.0 + expected_redshift) ** 2 * da(expected_redshift)
        )
        expected = generate_paired_log_mock(
            expected_redshift,
            da(expected_redshift),
            eta=expected_eta,
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            mode="random",
            seed=seed,
            rng=np.random.default_rng(noise_stream),
        )

        with tempfile.TemporaryDirectory() as directory:
            generator = QSOMockGenerator(
                {"H0": h0},
                specs,
                z_grid,
                str(Path(directory) / "mock"),
                dl_func=dimensionless_dl,
                da_func=da,
                sigma_ln_dl=0.3,
                sigma_ln_da=0.2,
                mode="random",
                seed=seed,
            )
        np.testing.assert_array_equal(generator.redshifts, expected_redshift)
        np.testing.assert_array_equal(generator.dl_data, expected.dl_observed)
        np.testing.assert_array_equal(generator.da_data, expected.da_observed)
        self.assertEqual(generator.paired_mock.seed, seed)


if __name__ == "__main__":
    unittest.main()
