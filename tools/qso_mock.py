import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import os

from .QSO_specifications import QSOSpecs

class QSOMockGenerator:
    """
    Generate synchronized mock luminosity and angular diameter distance data.

    The error budget is a natural-log scatter on each distance,

        sigma_ln_dl  from the UV--X-ray relation residual dispersion
        sigma_ln_da  from the SARM residual scatter

    obtained from a named scenario ('current', 'future', 'targeted') via
    QSOSpecs.error_budget, or passed explicitly. Noise is drawn
    multiplicatively, d_obs = d_true * exp(N(0, sigma_ln)), which is what a log
    scatter means and keeps distances positive: the 'current' budget puts
    sigma_ln_dl near 0.69, where additive Gaussian noise would routinely give
    negative distances.

    The saved error column is sigma_ln * d_observed, i.e. the fractional
    uncertainty on the measured distance, which is the form the likelihood
    reads back.

    Parameters
    ----------
    cosmoparams : dict
        Fiducial cosmology ('Omegam', 'H0'). Retained on the instance for
        reference; the error model is purely fractional and does not use it.
    specifications : dict
        Survey specifications ('n_QSO', 'z_min', 'z_max').
    z_in : array_like
        Redshift grid on which dl_func is tabulated.
    outpath : str
        Prefix for the saved '_dl_dataset.txt' / '_da_dataset.txt' files.
    dl_func, da_func : callable
        Fiducial luminosity and angular diameter distance functions, in Mpc.
    scenario : str, optional
        Named error budget, one of 'current', 'future', 'targeted'.
    sigma_ln_dl, sigma_ln_da : float, optional
        Explicit log scatters, overriding `scenario` when both are given.
    validate : bool, optional
        Redraw until the standardised log residuals have mean 0 and standard
        deviation 1 to within 0.01. This makes each catalogue much closer to
        its expectation than a real one would be, so the spread of recovered
        parameters across catalogues is not meaningful when it is on.
    """

    def __init__(self, cosmoparams, specifications, z_in, outpath, dl_func, da_func,
                 scenario='current', sigma_ln_dl=None, sigma_ln_da=None,
                 validate=True):
        print('Generating synchronized QSO mock datasets.')

        # 1. Generate a SINGLE redshift distribution for both
        self.redshifts = QSOSpecs.redshift_distribution(specifications)
        self.cosmoparams = cosmoparams

        # 2. Resolve the error budget
        if sigma_ln_dl is None or sigma_ln_da is None:
            self.sigma_ln_dl, self.sigma_ln_da = QSOSpecs.error_budget(scenario)
            self.scenario = scenario
        else:
            self.sigma_ln_dl, self.sigma_ln_da = float(sigma_ln_dl), float(sigma_ln_da)
            self.scenario = 'custom'
        print(f"Error budget [{self.scenario}]: "
              f"sigma_ln_dl={self.sigma_ln_dl:.3f}, sigma_ln_da={self.sigma_ln_da:.3f}")

        # 3. Setup theoretical functions
        # Both dl_func and da_func return distances in Mpc, so they are used as-is
        self.dl_theoretical = interp1d(z_in, dl_func(z_in))
        self.da_theoretical = da_func

        # 4. Draw both datasets against the same redshifts
        self.dl_data, self.dl_error = self._generate(
            self.dl_theoretical, self.sigma_ln_dl, "Luminosity", validate)
        self.da_data, self.da_error = self._generate(
            self.da_theoretical, self.sigma_ln_da, "Angular", validate)

        # 5. Save files
        self._save_data(outpath)

    def _generate(self, theory_func, sigma_ln, label, validate):
        """Draw one dataset with multiplicative log-normal noise."""
        truth = theory_func(self.redshifts)

        mu_res, sigma_res = 1.0, 3.0
        while True:
            data = truth * np.exp(np.random.normal(0.0, sigma_ln, len(truth)))
            # Residuals are Gaussian in log space, which is where the noise lives
            residuals = np.log(data / truth) / sigma_ln
            mu_res, sigma_res = norm.fit(residuals)
            if not validate:
                break
            if (np.abs(mu_res) <= 0.01) and (np.abs(sigma_res - 1) <= 0.01):
                break

        print(f"Validated {label} residuals: mu={mu_res:.4f}, sigma={sigma_res:.4f}")
        # Fractional uncertainty on the measured distance
        return data, sigma_ln * data

    def _save_data(self, outpath):
        # 1. Extract the directory from the outpath
        directory = os.path.dirname(outpath)

        # 2. Create the directory if it doesn't exist (and it's not the current dir)
        if directory and not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)

        # 3. Save dL
        np.savetxt(f"{outpath}_dl_dataset.txt",
                   np.column_stack([self.redshifts, self.dl_data, self.dl_error]),
                   delimiter=' ')
        # 4. Save dA
        np.savetxt(f"{outpath}_da_dataset.txt",
                   np.column_stack([self.redshifts, self.da_data, self.da_error]),
                   delimiter=' ')
        print(f"Datasets saved to {outpath}_dl/da_dataset.txt")

    def mock(self):
        return {
            'z': self.redshifts,
            'dl_value': self.dl_data,
            'dl_error': self.dl_error,
            'da_value': self.da_data,
            'da_error': self.da_error
        }
