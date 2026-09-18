import unittest
from unittest.mock import patch

import matplotlib as mpl
import numpy as np

from tools.mockchecker import MockChecker, _diagnostic_text, _log_sigma_error_bars


def current_metadata():
    return {
        "configuration": {
            "n_qso": 200,
            "z_min": 0.2,
            "z_max": 2.0,
            "error_scenario": "current",
        },
        "resolved_errors": {
            "name": "current",
            "sigma_ln_dl": 0.6907755278982137,
            "sigma_ln_da": 0.3,
            "errors_include_intrinsic": True,
        },
        "cosmology": {
            "model": "flat_lcdm",
            "H0_km_s_Mpc": 67.36,
            "omega_m": 0.314,
            "omega_k": 0.0,
            "w0": -1.0,
        },
        "ddr": {
            "model": "M1",
            "alpha": -0.075,
            "beta": 0.0,
        },
        "mock_generation": {
            "mode": "random",
            "master_seed": 27,
        },
    }


def schema_v2_metadata(uvx_option="current"):
    metadata = current_metadata()
    metadata["schema_version"] = 2
    metadata["errors_include_intrinsic"] = True
    metadata["configuration"]["rho"] = -0.25
    metadata["uvx_errors"] = {
        "option": uvx_option,
        "gamma": 0.60,
        "gamma_used": uvx_option != "custom",
        "sigma_uvx_dex": 0.24 if uvx_option == "current" else None,
        "custom_sigma_ln_dl": 0.31 if uvx_option == "custom" else None,
        "delta_dex": 0.21 if uvx_option == "lusso2020_flux_table" else None,
    }
    metadata["sarm_errors"] = {
        "option": "current",
        "sigma_ln_da": 0.30,
    }
    metadata.pop("resolved_errors")
    if uvx_option == "lusso2020_flux_table":
        metadata["diagnostic_uncertainty_summary"] = {
            "sigma_ln_dl": {
                "min": 0.31,
                "median": 0.42,
                "max": 0.58,
            }
        }
    return metadata


class MockCheckerTests(unittest.TestCase):
    def test_log_sigma_error_bars_are_asymmetric_linear_68_percent_intervals(self):
        error_bars = _log_sigma_error_bars(
            np.array([100.0]), np.array([np.log(2.0)])
        )

        np.testing.assert_allclose(error_bars, [[50.0], [100.0]])

    def test_diagnostic_text_uses_mock_name_and_mathematical_symbols(self):
        title, lines = _diagnostic_text(
            current_metadata(),
            "M1alpha-0.075beta0.0nQSO200",
        )
        joined = "\n".join(lines)
        self.assertEqual(title, "M1alpha-0.075beta0.0nQSO200")
        for text in (
            r"$N_{\mathrm{QSO}}=200$",
            r"$\sigma_{\ln D_L}=0.6908$",
            r"$\sigma_{\ln D_A}=0.3000$",
            r"$\rho=0$",
            r"flat $\Lambda$CDM",
            r"$H_0=67.36$",
            r"$\Omega_m=0.314$",
            r"$\eta(z)=1+\alpha$",
            r"$\alpha=-0.075$",
        ):
            self.assertIn(text, joined)

    def test_schema_v2_diagnostic_text_separates_named_uvx_and_sarm_models(self):
        _, lines = _diagnostic_text(schema_v2_metadata(), "named-errors")
        joined = "\n".join(lines)

        for text in (
            "UV-X = current",
            r"$\gamma=0.60$",
            r"$\sigma_{\ln D_L}=0.6908$",
            "SARM = current",
            r"$\sigma_{\ln D_A}=0.3000$",
            r"$\rho=-0.25$",
            "intrinsic included",
        ):
            self.assertIn(text, joined)

    def test_schema_v2_diagnostic_text_reports_table_distance_error_range(self):
        metadata = schema_v2_metadata("lusso2020_flux_table")

        _, lines = _diagnostic_text(metadata, "table-errors")
        joined = "\n".join(lines)

        for text in (
            "UV-X = lusso2020_flux_table",
            r"$\gamma=0.60$",
            r"$\delta=0.21\ \mathrm{dex}$",
            r"$\sigma_{\ln D_L}=0.4200\ [0.3100, 0.5800]$",
            "SARM = current",
            r"$\sigma_{\ln D_A}=0.3000$",
        ):
            self.assertIn(text, joined)

    def test_schema_v2_custom_diagnostic_marks_gamma_unused(self):
        _, lines = _diagnostic_text(schema_v2_metadata("custom"), "custom-errors")
        joined = "\n".join(lines)

        self.assertIn("UV-X = custom", joined)
        self.assertIn(r"$\gamma$ not used", joined)
        self.assertNotIn(r"$\gamma=0.60$", joined)
        self.assertIn(r"$\sigma_{\ln D_L}=0.3100$", joined)

    def test_diagnostic_text_describes_cosmology_and_ddr_variants(self):
        cases = (
            (
                "lcdm",
                "M2",
                r"$\Omega_k=0.12$",
                r"$\eta(z)=1+\alpha+\beta z$",
                r"$\alpha=-0.075$",
                r"$\beta=0.01$",
            ),
            (
                "flat_wcdm",
                "M3",
                r"$w_0=-0.8$",
                r"$\eta(z)=(1+z)^\alpha$",
                r"$\alpha=-0.075$",
            ),
        )
        for cosmology_model, ddr_model, *expected_text in cases:
            with self.subTest(cosmology_model=cosmology_model, ddr_model=ddr_model):
                metadata = current_metadata()
                metadata["cosmology"]["model"] = cosmology_model
                metadata["cosmology"]["omega_k"] = 0.12
                metadata["cosmology"]["w0"] = -0.8
                metadata["ddr"]["model"] = ddr_model
                metadata["ddr"]["beta"] = 0.01

                _, lines = _diagnostic_text(metadata, "variant")
                joined = "\n".join(lines)
                for text in expected_text:
                    self.assertIn(text, joined)

    def test_diagnostic_text_reports_both_custom_natural_log_scatters(self):
        metadata = current_metadata()
        metadata["configuration"]["error_scenario"] = "custom"
        metadata["resolved_errors"] = {
            "name": "custom",
            "sigma_ln_dl": 0.31,
            "sigma_ln_da": 0.22,
            "errors_include_intrinsic": False,
        }

        _, lines = _diagnostic_text(metadata, "custom-errors")
        header = lines[1]

        self.assertIn("Errors = custom", header)
        self.assertIn(r"$\sigma_{\ln D_L}=0.3100$", header)
        self.assertIn(r"$\sigma_{\ln D_A}=0.2200$", header)

    def test_diagnostic_text_marks_asimov_mocks_deterministic_without_seed(self):
        metadata = current_metadata()
        metadata["mock_generation"] = {"mode": "asimov"}

        _, lines = _diagnostic_text(metadata, "asimov")
        joined = "\n".join(lines)
        self.assertIn("mode = Asimov", joined)
        self.assertIn("deterministic", joined)
        self.assertNotIn("seed =", joined)

    def test_paired_diagnostic_saves_png_without_mutating_mock(self):
        import tempfile
        from pathlib import Path

        from tools.mockchecker import save_paired_mock_diagnostic
        from tools.qso_mock import generate_paired_log_mock

        z = np.array([0.2, 0.5, 1.0])
        da = np.array([500.0, 900.0, 1200.0])
        mock = generate_paired_log_mock(
            z,
            da,
            eta=0.925,
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            mode="random",
            seed=7,
        )
        original_fields = {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in vars(mock).items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.png"
            tracked_rc = (
                "font.family",
                "mathtext.fontset",
                "text.usetex",
                "xtick.top",
                "ytick.right",
            )
            before = {name: mpl.rcParams[name] for name in tracked_rc}
            returned = save_paired_mock_diagnostic(
                path,
                mock,
                metadata=current_metadata(),
                title="M1alpha-0.075beta0.0nQSO200",
            )
            self.assertEqual(returned, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            for name, value in before.items():
                self.assertEqual(mpl.rcParams[name], value)
        for name, original in original_fields.items():
            current = getattr(mock, name)
            if isinstance(original, np.ndarray):
                np.testing.assert_array_equal(current, original)
            else:
                self.assertEqual(current, original)

    def test_paired_diagnostic_uses_mathematical_labels_and_requested_title(self):
        from pathlib import Path

        from tools.mockchecker import save_paired_mock_diagnostic
        from tools.qso_mock import generate_paired_log_mock

        mock = generate_paired_log_mock(
            np.array([0.2, 0.5, 1.0]),
            np.array([500.0, 900.0, 1200.0]),
            eta=0.925,
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            mode="random",
            seed=7,
        )
        captured_figures = []

        def capture_figure(figure, *args, **kwargs):
            captured_figures.append(figure)

        with patch(
            "matplotlib.figure.Figure.savefig",
            autospec=True,
            side_effect=capture_figure,
        ):
            returned = save_paired_mock_diagnostic(
                Path("diagnostic.png"),
                mock,
                metadata=current_metadata(),
                title="M1alpha-0.075beta0.0nQSO200",
            )

        self.assertEqual(returned, Path("diagnostic.png"))
        self.assertEqual(len(captured_figures), 1)
        figure = captured_figures[0]
        self.assertEqual(figure._suptitle.get_text(), "M1alpha-0.075beta0.0nQSO200")
        axis_text = {axis.get_xlabel() for axis in figure.axes}
        self.assertIn(r"$z$", axis_text)
        self.assertIn(
            r"$\ln(D_L^\mathrm{obs}/D_L^\mathrm{true})/\sigma_{\ln D_L}$",
            axis_text,
        )
        self.assertIn(
            r"$\ln(D_A^\mathrm{obs}/D_A^\mathrm{true})/\sigma_{\ln D_A}$",
            axis_text,
        )

    def test_paired_diagnostic_renders_error_bars_for_both_distance_probes(self):
        from pathlib import Path

        from tools.mockchecker import save_paired_mock_diagnostic
        from tools.qso_mock import generate_paired_log_mock

        mock = generate_paired_log_mock(
            np.array([0.2, 0.5, 1.0]),
            np.array([500.0, 900.0, 1200.0]),
            eta=0.925,
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            mode="random",
            seed=7,
        )
        captured_figures = []

        def capture_figure(figure, *args, **kwargs):
            captured_figures.append(figure)

        with patch(
            "matplotlib.figure.Figure.savefig",
            autospec=True,
            side_effect=capture_figure,
        ):
            save_paired_mock_diagnostic(Path("diagnostic.png"), mock)

        figure = captured_figures[0]
        self.assertEqual(len(figure.axes[0].containers), 1)
        self.assertEqual(len(figure.axes[1].containers), 1)

    def test_paired_diagnostic_uses_clean_style_inside_hostile_outer_context(self):
        from pathlib import Path

        from tools.mockchecker import save_paired_mock_diagnostic
        from tools.qso_mock import generate_paired_log_mock

        mock = generate_paired_log_mock(
            np.array([0.2, 0.5, 1.0]),
            np.array([500.0, 900.0, 1200.0]),
            eta=0.925,
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            mode="random",
            seed=7,
        )
        hostile = {
            "figure.facecolor": "black",
            "axes.facecolor": "black",
            "savefig.facecolor": "black",
            "text.color": "magenta",
            "axes.labelcolor": "magenta",
            "axes.titlecolor": "magenta",
            "xtick.color": "magenta",
            "ytick.color": "magenta",
            "font.family": "sans-serif",
            "font.size": 23.0,
            "axes.labelsize": 23.0,
            "axes.titlesize": 23.0,
            "legend.fontsize": 23.0,
            "xtick.labelsize": 23.0,
            "ytick.labelsize": 23.0,
        }
        captured_figures = []

        def capture_figure(figure, *args, **kwargs):
            captured_figures.append(figure)
            self.assertEqual(mpl.rcParams["savefig.facecolor"], "white")

        global_before = {name: mpl.rcParams[name] for name in hostile}
        with mpl.rc_context(hostile):
            outer_before = {name: mpl.rcParams[name] for name in hostile}
            with patch(
                "matplotlib.figure.Figure.savefig",
                autospec=True,
                side_effect=capture_figure,
            ):
                save_paired_mock_diagnostic(
                    Path("diagnostic.png"), mock, metadata=current_metadata()
                )
            self.assertEqual(
                {name: mpl.rcParams[name] for name in hostile},
                outer_before,
            )
        self.assertEqual({name: mpl.rcParams[name] for name in hostile}, global_before)

        figure = captured_figures[0]
        summary = next(
            text for text in figure.texts if r"$N_{\mathrm{QSO}}=200$" in text.get_text()
        )
        self.assertEqual(figure.get_facecolor(), mpl.colors.to_rgba("white"))
        self.assertEqual(figure._suptitle.get_color(), "black")
        self.assertEqual(figure._suptitle.get_fontfamily(), ["serif"])
        self.assertEqual(summary.get_color(), "black")
        self.assertEqual(summary.get_fontsize(), 10.5)
        self.assertEqual(summary.get_fontfamily(), ["serif"])
        for axis in figure.axes:
            self.assertEqual(axis.get_facecolor(), mpl.colors.to_rgba("white"))
            self.assertEqual(axis.xaxis.label.get_color(), "black")
            self.assertEqual(axis.title.get_color(), "black")
            self.assertEqual(axis.xaxis.label.get_fontsize(), 10.0)
            self.assertEqual(axis.title.get_fontsize(), 12.0)
            self.assertEqual(axis.get_xticklabels()[0].get_color(), "black")
            self.assertEqual(axis.get_xticklabels()[0].get_fontsize(), 10.0)
            self.assertEqual(axis.get_yticklabels()[0].get_color(), "black")
            self.assertEqual(axis.get_yticklabels()[0].get_fontsize(), 10.0)
            if axis.get_legend() is not None:
                self.assertEqual(axis.get_legend().get_texts()[0].get_fontsize(), 10.0)
                np.testing.assert_allclose(
                    axis.get_legend().get_texts()[0].get_color(),
                    mpl.colors.to_rgba("black"),
                )

    def test_paired_diagnostic_uses_mathtext_without_external_latex(self):
        from pathlib import Path

        from matplotlib.backends.backend_agg import FigureCanvasAgg

        from tools.mockchecker import save_paired_mock_diagnostic
        from tools.qso_mock import generate_paired_log_mock

        mock = generate_paired_log_mock(
            np.array([0.2, 0.5, 1.0]),
            np.array([500.0, 900.0, 1200.0]),
            eta=0.925,
            sigma_ln_dl=0.3,
            sigma_ln_da=0.2,
            mode="random",
            seed=7,
        )
        captured_style = []

        def draw_figure(figure, *args, **kwargs):
            self.assertFalse(mpl.rcParams["text.usetex"])
            captured_style.append(
                (mpl.rcParams["text.usetex"], mpl.rcParams["mathtext.fontset"])
            )
            FigureCanvasAgg(figure).draw()

        with patch(
            "matplotlib.figure.Figure.savefig",
            autospec=True,
            side_effect=draw_figure,
        ):
            save_paired_mock_diagnostic(Path("diagnostic.png"), mock)

        self.assertEqual(captured_style, [(False, "cm")])

    def test_standardized_residuals_use_elementwise_errors(self):
        residuals = MockChecker.standardized_residuals(
            data=np.array([11.0, 18.0]),
            fiducial=np.array([10.0, 20.0]),
            error=np.array([1.0, 2.0]),
        )
        np.testing.assert_array_equal(residuals, np.array([1.0, -1.0]))

    def test_standardized_residuals_reject_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "broadcast"):
            MockChecker.standardized_residuals([1.0, 2.0], [1.0], [0.1, 0.2, 0.3])

    def test_visualize_false_does_not_create_plots(self):
        class RecordingChecker(MockChecker):
            plotted = 0

            def histogram(self, name):
                self.plotted += 1

            def plot_mock(self, name):
                self.plotted += 1

        checker = RecordingChecker(
            x=np.array([0.1, 0.2]),
            data=np.array([1.0, 1.0]),
            error=np.array([0.1, 0.1]),
            fiducial=np.array([1.0, 1.0]),
            threshold=2.0,
            visualize=False,
        )
        self.assertEqual(checker.plotted, 0)


if __name__ == "__main__":
    unittest.main()
