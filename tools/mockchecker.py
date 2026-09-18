from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


_DIAGNOSTIC_STYLE = Path(__file__).with_name("qso_gp_mock_matplotlibrc")
_DIAGNOSTIC_STYLE_OVERRIDES = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.size": 10.0,
    "axes.labelsize": 10.0,
    "axes.titlesize": 12.0,
    "legend.fontsize": 10.0,
    "legend.labelcolor": "black",
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
}


def _number(value, digits=4):
    return f"{float(value):.{digits}g}"


def _log_sigma_error_bars(distance, sigma_ln_distance):
    """Return lower/upper linear-distance bars for a log-space 68% width."""
    distance = np.asarray(distance, dtype=float)
    sigma_ln_distance = np.asarray(sigma_ln_distance, dtype=float)
    return np.vstack((
        distance * (1.0 - np.exp(-sigma_ln_distance)),
        distance * (np.exp(sigma_ln_distance) - 1.0),
    ))


def _diagnostic_style():
    style = mpl.rcParamsDefault.copy()
    style.update(
        mpl.rc_params_from_file(
            _DIAGNOSTIC_STYLE,
            use_default_template=False,
        )
    )
    style.update(_DIAGNOSTIC_STYLE_OVERRIDES)
    # MathText preserves Computer Modern mathematical notation without
    # requiring external LaTeX processes or a writable TeX cache.
    style["text.usetex"] = False
    return style


def _diagnostic_text(
    metadata: Mapping[str, Any] | None,
    mock_name: str,
) -> tuple[str, tuple[str, ...]]:
    if metadata is None:
        return mock_name, ()

    config = metadata["configuration"]
    cosmology = metadata["cosmology"]
    ddr = metadata["ddr"]
    generation = metadata["mock_generation"]

    mode = generation["mode"]
    if mode == "random":
        mode_text = f"mode = random | seed = {generation['master_seed']}"
    else:
        mode_text = "mode = Asimov | deterministic"

    line1 = (
        rf"$N_{{\mathrm{{QSO}}}}={int(config['n_qso'])}$"
        rf" | {mode_text}"
        rf" | $z\in[{_number(config['z_min'])},"
        rf" {_number(config['z_max'])}]$"
    )
    if metadata.get("schema_version") == 2:
        uvx = metadata["uvx_errors"]
        sarm = metadata["sarm_errors"]
        summary = metadata.get("diagnostic_uncertainty_summary", {}).get(
            "sigma_ln_dl"
        )
        intrinsic_text = (
            "intrinsic included"
            if metadata["errors_include_intrinsic"]
            else "intrinsic separate"
        )
        line2 = f"UV-X = {uvx['option']}"
        if uvx["gamma_used"]:
            line2 += rf" | $\gamma={float(uvx['gamma']):.2f}$"
        else:
            line2 += r" | $\gamma$ not used"
        if uvx["option"] == "lusso2020_flux_table":
            if summary is None:
                raise ValueError(
                    "table-mode diagnostic requires a sigma_ln_dl "
                    "realization summary"
                )
            line2 += (
                rf" | $\delta={float(uvx['delta_dex']):.2f}"
                r"\ \mathrm{dex}$"
                rf" | $\sigma_{{\ln D_L}}={float(summary['median']):.4f}"
                rf"\ [{float(summary['min']):.4f},"
                rf" {float(summary['max']):.4f}]$"
            )
        else:
            if summary is not None:
                sigma_ln_dl = float(summary["median"])
            elif uvx["option"] == "custom":
                sigma_ln_dl = float(uvx["custom_sigma_ln_dl"])
            else:
                gamma = float(uvx["gamma"])
                sigma_ln_dl = (
                    np.log(10.0)
                    * float(uvx["sigma_uvx_dex"])
                    / (2.0 * (1.0 - gamma))
                )
            line2 += (
                rf" | $\sigma_{{\ln D_L}}="
                rf"{sigma_ln_dl:.4f}$"
            )
        line2 += f" | {intrinsic_text}"
        line_sarm = (
            f"SARM = {sarm['option']}"
            rf" | $\sigma_{{\ln D_A}}={float(sarm['sigma_ln_da']):.4f}$"
            rf" | $\rho={_number(config.get('rho', 0.0))}$"
        )
    else:
        errors = metadata["resolved_errors"]
        intrinsic_text = (
            "intrinsic included"
            if errors["errors_include_intrinsic"]
            else "intrinsic separate"
        )
        line2 = (
            f"Errors = {errors['name']}"
            rf" | $\sigma_{{\ln D_L}}={float(errors['sigma_ln_dl']):.4f}$"
            rf" | $\sigma_{{\ln D_A}}={float(errors['sigma_ln_da']):.4f}$"
            rf" | $\rho={_number(config.get('rho', 0.0))}$"
            f" | {intrinsic_text}"
        )
        line_sarm = None

    model = cosmology["model"]
    if model == "flat_lcdm":
        cosmology_text = r"flat $\Lambda$CDM"
    elif model == "lcdm":
        cosmology_text = (
            r"$\Lambda$CDM"
            rf" | $\Omega_k={_number(cosmology['omega_k'])}$"
        )
    else:
        cosmology_text = (
            r"flat $w$CDM"
            rf" | $w_0={_number(cosmology['w0'])}$"
        )
    line3 = (
        f"Cosmology = {cosmology_text}"
        rf" | $H_0={_number(cosmology['H0_km_s_Mpc'])}$"
        rf" | $\Omega_m={_number(cosmology['omega_m'])}$"
    )

    ddr_model = ddr["model"]
    alpha = _number(ddr["alpha"])
    beta = _number(ddr["beta"])
    if ddr_model == "M1":
        equation = r"$\eta(z)=1+\alpha$" + rf" | $\alpha={alpha}$"
    elif ddr_model == "M2":
        equation = (
            r"$\eta(z)=1+\alpha+\beta z$"
            rf" | $\alpha={alpha}$ | $\beta={beta}$"
        )
    else:
        equation = (
            r"$\eta(z)=(1+z)^\alpha$"
            rf" | $\alpha={alpha}$"
        )
    line4 = f"DDR = {ddr_model} | {equation}"
    lines = (line1, line2, line3, line4)
    if line_sarm is not None:
        lines = (line1, line2, line_sarm, line3, line4)
    return mock_name, lines


def save_paired_mock_diagnostic(
    path,
    mock,
    *,
    metadata=None,
    title=None,
):
    """Save DL/DA truth comparisons and log-residual histograms without selection."""
    path = Path(path)
    dl_residual = np.log(mock.dl_observed / mock.dl_true)
    da_residual = np.log(mock.da_observed / mock.da_true)
    style = _diagnostic_style()

    mock_name = title or path.stem.removesuffix("_diagnostic")
    figure_title, summary_lines = _diagnostic_text(metadata, mock_name)

    with mpl.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
        fig.suptitle(figure_title, fontsize=17, y=0.985)
        if summary_lines:
            fig.text(
                0.5,
                0.945,
                "\n".join(summary_lines),
                ha="center",
                va="top",
                fontsize=10.5,
                linespacing=1.35,
            )
        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            bottom=0.09,
            top=0.73,
            hspace=0.36,
            wspace=0.24,
        )

        axes[0, 0].plot(mock.redshift, mock.dl_true, color="#8E001C", label="truth")
        axes[0, 0].errorbar(
            mock.redshift,
            mock.dl_observed,
            yerr=_log_sigma_error_bars(mock.dl_observed, mock.sigma_ln_dl),
            fmt="o",
            color="#FFB300",
            markersize=3.5,
            elinewidth=0.55,
            capsize=0,
            alpha=0.42,
            label="mock: assigned 68% interval",
        )
        axes[0, 0].set(
            xlabel=r"$z$",
            ylabel=r"$D_L\,[\mathrm{Mpc}]$",
            title=r"Luminosity distance $D_L$",
        )
        axes[0, 0].legend()

        axes[0, 1].plot(mock.redshift, mock.da_true, color="#8E001C", label="truth")
        axes[0, 1].errorbar(
            mock.redshift,
            mock.da_observed,
            yerr=_log_sigma_error_bars(mock.da_observed, mock.sigma_ln_da),
            fmt="o",
            color="#FFB300",
            markersize=3.5,
            elinewidth=0.55,
            capsize=0,
            alpha=0.42,
            label="mock: assigned 68% interval",
        )
        axes[0, 1].set(
            xlabel=r"$z$",
            ylabel=r"$D_A\,[\mathrm{Mpc}]$",
            title=r"Angular-diameter distance $D_A$",
        )
        axes[0, 1].legend()

        axes[1, 0].hist(
            dl_residual / np.where(mock.sigma_ln_dl > 0, mock.sigma_ln_dl, 1.0),
            bins="auto",
            color="#2a78d6",
        )
        axes[1, 0].set(
            xlabel=r"$\ln(D_L^\mathrm{obs}/D_L^\mathrm{true})/\sigma_{\ln D_L}$",
            title=r"Luminosity-distance log residuals",
        )

        axes[1, 1].hist(
            da_residual / np.where(mock.sigma_ln_da > 0, mock.sigma_ln_da, 1.0),
            bins="auto",
            color="#008300",
        )
        axes[1, 1].set(
            xlabel=r"$\ln(D_A^\mathrm{obs}/D_A^\mathrm{true})/\sigma_{\ln D_A}$",
            title=r"Angular-distance log residuals",
        )

        for axis in axes.flat:
            for spine in axis.spines.values():
                spine.set_linewidth(1.2)
            axis.tick_params(which="both", direction="in", top=True, right=True)

        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return path


class MockChecker:
    """
    Validate and visualize mock data quality.

    This class performs quality checks on generated mock data by comparing
    to fiducial model predictions, computing outliers, and creating diagnostic
    plots to verify the data follows expected statistical distributions.

    Parameters
    ----------
    x : array_like
        Independent variable (e.g., redshift)
    data : array_like
        Mock data values
    error : array_like or float
        Observational errors (can be array or scalar)
    fiducial : array_like
        Fiducial model predictions
    threshold : float
        Outlier threshold in units of sigma
    visualize : bool, optional
        Whether to create visualizations (default: False)
    dataname : str, optional
        Name for plots and reports (default: 'test')

    Attributes
    ----------
    outliers : dict
        Dictionary containing outlier information
    """

    def __init__(self, x=None, data=None, error=None, fiducial=None,
                 threshold=None, visualize=False, dataname='test'):

        self.x         = x
        self.data      = data
        self.error     = error
        self.fiducial  = fiducial
        self.threshold = threshold



        print('-----------------------------------REPORT ON '+dataname+'-----------------------------------')

        self.outliers = self.get_outliers()

        print('Number of outliers: '+str(len(self.outliers['x'])))
        print('Fraction of data: '+str(100*len(self.outliers['x'])/len(self.x))+'%')
        if len(self.outliers['x'])>0:
            print('Maximum distance from fiducial (in sigma): '+str(np.sort(self.outliers['distance'])[-1]))


        if visualize:
            self.histogram(dataname)
            self.plot_mock(dataname)


    def histogram(self,name):
        hist = self.standardized_residuals(self.data, self.fiducial, self.error)

        mu,sigma = norm.fit(hist)
        x = np.linspace(-10,10,100)

        limit = max(np.abs(min(hist)),max(hist))

        plt.figure(figsize=(10, 8))
        plt.title(name+' histogram', fontsize=20)
        plt.hist(hist, density=True, bins='auto', edgecolor='black', linewidth=2)  # `density=False` would make counts
        plt.plot(x,norm.pdf(x,mu,sigma),ls='-',linewidth=3,color='#FFB300',label=r'$\mu=$'+str(mu)+r' $\sigma=$'+str(sigma))
        plt.ylabel('Counts', fontsize=20)
        plt.xlabel(r'(Data-fiducial)/$\sigma$', fontsize=20)
        plt.axvline(0,ls='--',lw=3,color='#8E001C')
        plt.xlim([-3*limit,3*limit])
        plt.legend(loc='upper right')
        plt.show()

        return None

    def plot_mock(self,name):

        plt.figure(figsize=(10, 8))
        plt.title(name+' mock',fontsize=20)
        plt.plot(self.x,self.fiducial,label=r'Fiducial',ls='-',color='#8E001C')
        plt.errorbar(self.x,self.data,yerr=self.error,label='data',ls='',marker='*',color='#FFB300')
        plt.errorbar(self.outliers['x'],self.outliers['data'],
                     label=r' '+str(self.threshold)+r'$\sigma$ outliers ('+str(100*len(self.outliers['x'])/len(self.x))+'% of total)',
                     ls='',marker='x',markersize=10,color='black')
        plt.xlabel(r'$x$',fontsize=20)
        plt.ylabel(r'$y(x)$',fontsize=20)
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left",fontsize=20)
        plt.show()


        
    def get_outliers(self):

        outliers = {}


        distance = np.abs(
            self.standardized_residuals(self.data, self.fiducial, self.error)
        )

        def condition(x): return x > self.threshold

        outliers['x']        = np.array([self.x[idx] for idx, element in enumerate(distance) if condition(element)])
        outliers['data']     = np.array([self.data[idx] for idx, element in enumerate(distance) if condition(element)])
        errors = np.broadcast_to(np.asarray(self.error), np.asarray(self.data).shape)
        outliers['error'] = np.array(
            [errors[idx] for idx, element in enumerate(distance) if condition(element)]
        )
        outliers['distance'] = np.array([element for  element in distance if condition(element)])

        return outliers




    




    @staticmethod
    def standardized_residuals(data, fiducial, error):
        """Return elementwise standardized residuals with validated broadcasting."""
        try:
            data, fiducial, error = np.broadcast_arrays(
                np.asarray(data, dtype=float),
                np.asarray(fiducial, dtype=float),
                np.asarray(error, dtype=float),
            )
        except ValueError as exc:
            raise ValueError("data, fiducial, and error must broadcast together") from exc
        if np.any(error <= 0):
            raise ValueError("error must be strictly positive")
        return (data - fiducial) / error
