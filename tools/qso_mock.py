import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import os

from .QSO_specifications import QSOSpecs
from .constants import SPEED_OF_LIGHT_KMS

class QSOMockGenerator:
    """
    Generate synchronized mock luminosity and angular diameter distance data.
    """

    def __init__(self, cosmoparams, specifications, z_in, outpath, dl_func, da_func, 
                 sigma_uvx_fid=0.05, beta_dl_meas=0.05,
                 sigma_sarm_fid=0.05, beta_da_meas=0.05):
        print('Generating synchronized QSO mock datasets.')

        # 1. Generate a SINGLE redshift distribution for both
        self.redshifts = QSOSpecs.redshift_distribution(specifications)
        n_points = len(self.redshifts)
        hubble_dist = SPEED_OF_LIGHT_KMS / cosmoparams['H0']

        # 2. Setup theoretical functions and errors
        # Luminosity Distance setup
        self.dl_theoretical = interp1d(z_in, hubble_dist * dl_func(z_in))
        dl_err_meas = beta_dl_meas * (hubble_dist*dl_func(self.redshifts))
        dl_err_intrinsic = sigma_uvx_fid * (hubble_dist*dl_func(self.redshifts))
        self.dl_error = np.full(n_points, np.sqrt(dl_err_meas**2 + dl_err_intrinsic**2))

        # Angular Distance setup
        # Note: assuming da_func already returns Mpc or is handled similarly
        self.da_theoretical = da_func 
        da_err_meas = beta_da_meas * da_func(self.redshifts)
        da_err_intrinsic = sigma_sarm_fid * da_func(self.redshifts)
        self.da_error = np.full(n_points, np.sqrt(da_err_meas**2 + da_err_intrinsic**2))

        # 3. Unified Validation Loop
        # We ensure BOTH datasets pass the N(0,1) residual test
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
