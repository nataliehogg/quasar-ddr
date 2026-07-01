import numpy as np
from cobaya.likelihood import Likelihood

from tools.theory import Theory
from tools.utilities import Utilities

class DDRlike(Likelihood):
    """
    Cosmology-independent Cobaya likelihood for Distance Duality Relation
    (DDR) violation constraints using alpha and beta parameters.
    """
    datapath: str = './mock_data/test'
    model: str = 'M1'  # DDR violation model: 'M1', 'M2' or 'M3'
    sample_sigma_int: bool = True  # whether sigma_int is sampled (must match the config)

    def initialize(self):
        """Load mock data and identify model selection once at startup."""

        # Model string is passed straight to Theory, which validates it
        self.EMmodel = self.model

        # This print statement will only appear once when Cobaya starts
        print(f"--- [DDRlike] Model Selected: {self.EMmodel} ---")
        print(f"[DDRlike] Loading data from: {self.datapath}")

        # Load mock data
        dl_data = np.loadtxt(self.datapath + '_dl_dataset.txt')
        da_data = np.loadtxt(self.datapath + '_da_dataset.txt')

        self.z = dl_data[:, 0]
        self.dl = dl_data[:, 1]
        self.error_dl = dl_data[:, 2]
        self.da = da_data[:, 1]
        self.error_da = da_data[:, 2]

        # Calculate observational eta and statistical errors
        self.eta_obs = self.dl / (self.da * (1 + self.z)**2)
        self.stat_error_eta = self.eta_obs * np.sqrt((self.error_dl / self.dl)**2 + (self.error_da / self.da)**2)

    def get_requirements(self):
        """Require only the parameters this model actually samples.

        Uses the same MODEL_PARAMS mapping as Utilities.sampler_config, so the
        required parameters always match those declared in the config (M1/M3:
        alpha; M2: alpha, beta), plus sigma_int when it is being sampled.
        """
        names = list(Utilities.MODEL_PARAMS[self.model])
        if self.sample_sigma_int:
            names.append('sigma_int')
        return {name: None for name in names}

    def logp(self, alpha=0, beta=0, sigma_int=0, **params_values):
        """Compute log-likelihood for DDR parameters alpha/beta and intrinsic scatter."""
        
        # Calculate theoretical eta using updated nomenclature
        TH = Theory(
            z=self.z, 
            d_C=None, 
            d_A=None, 
            H=None, 
            Om=None, 
            H0=None, 
            alpha=alpha, 
            beta=beta,   
            model=self.EMmodel
        )
        eta_th = TH.eta

        # Calculate variance including intrinsic scatter
        variance_eta = self.stat_error_eta**2 + (self.eta_obs * sigma_int)**2

        # Log-likelihood computation
        residuals = eta_th - self.eta_obs
        chi2_term = np.sum((residuals**2) / variance_eta)
        norm_term = np.sum(np.log(2 * np.pi * variance_eta))

        return -0.5 * (chi2_term + norm_term)
