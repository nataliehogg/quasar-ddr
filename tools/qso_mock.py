import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import os

from .QSO_specifications import QSOSpecs

class QSOMockGenerator:
    """
    Generate synchronized mock luminosity and angular diameter distance data.

    Errors are fractional, following the error budget of the paper: the
    per-object uncertainty on each distance is the quadrature sum of a
    measurement term and an intrinsic-scatter term, both expressed as a
    fraction of the distance itself,

        (sigma_dL / dL)^2 = sigma_dl_meas^2 + sigma_uvx^2
        (sigma_dA / dA)^2 = sigma_da_meas^2 + sigma_sarm^2

    where sigma_uvx is the residual dispersion of the UV--X-ray relation and
    sigma_sarm the residual BLR-modelling scatter of the SARM method. The
    stored errors are absolute (Mpc), since that is what the likelihood reads.

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
    sigma_dl_meas, sigma_da_meas : float
        Fractional measurement uncertainty on d_L and d_A.
    sigma_uvx_fid, sigma_sarm_fid : float
        Fractional intrinsic scatter on d_L (UV--X-ray) and d_A (SARM).
    """

    def __init__(self, cosmoparams, specifications, z_in, outpath, dl_func, da_func,
                 sigma_dl_meas=0.05, sigma_uvx_fid=0.05,
                 sigma_da_meas=0.05, sigma_sarm_fid=0.05):
        print('Generating synchronized QSO mock datasets.')

        # 1. Generate a SINGLE redshift distribution for both
        self.redshifts = QSOSpecs.redshift_distribution(specifications)
        self.cosmoparams = cosmoparams

        # 2. Setup theoretical functions and errors
        # Both dl_func and da_func return distances in Mpc, so they are used
        # as-is; the fractional errors below set the absolute error scale.
        # Luminosity Distance setup
        self.dl_theoretical = interp1d(z_in, dl_func(z_in))
        dl_fiducial = self.dl_theoretical(self.redshifts)
        self.dl_error = dl_fiducial * np.sqrt(sigma_dl_meas**2 + sigma_uvx_fid**2)

        # Angular Distance setup
        self.da_theoretical = da_func
        da_fiducial = self.da_theoretical(self.redshifts)
        self.da_error = da_fiducial * np.sqrt(sigma_da_meas**2 + sigma_sarm_fid**2)

        # 3. Unified Validation Loop
        # We ensure BOTH datasets pass the N(0,1) residual test
        n_points = len(self.redshifts)
        self.dl_data = np.zeros(n_points)
        self.da_data = np.zeros(n_points)

        self._generate_validated_data(self.dl_theoretical, self.dl_error, "Luminosity", is_dl=True)
        self._generate_validated_data(self.da_theoretical, self.da_error, "Angular", is_dl=False)

        # 4. Save files
        self._save_data(outpath)

    def _generate_validated_data(self, theory_func, error_arr, label, is_dl=True):
        mu_res, sigma_res = 1, 3
        data = np.zeros(len(self.redshifts))

        while (np.abs(mu_res) > 0.01) or (np.abs(sigma_res - 1) > 0.01):
            # Vectorized generation is faster than list comprehension
            noise = np.random.normal(0.0, error_arr)
            theory_vals = np.array([theory_func(zi) for zi in self.redshifts])
            data = theory_vals + noise

            residuals = (data - theory_vals) / error_arr
            mu_res, sigma_res = norm.fit(residuals)

        if is_dl: self.dl_data = data
        else: self.da_data = data
        print(f"Validated {label} residuals: mu={mu_res:.4f}, sigma={sigma_res:.4f}")

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
