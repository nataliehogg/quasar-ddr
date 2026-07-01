import numpy as np
from cobaya.likelihood import Likelihood
from tools.theory import Theory

class DDRlike(Likelihood):
    """
    Cosmology-independent Cobaya likelihood for DDR constraints.
    Samples sigma_uvx and sigma_sarm as FRACTIONAL intrinsic scatters.
    """
    datapath: str = './mock_data/test'
    model: int = 4  

    def initialize(self):
        model_map = {1: 'M1', 2: 'M2', 3: 'M3'}
        self.EMmodel = model_map.get(self.model, 'M1')
        
        dl_data = np.loadtxt(self.datapath + '_dl_dataset.txt')
        da_data = np.loadtxt(self.datapath + '_da_dataset.txt')

        self.z = dl_data[:, 0]
        self.dl = dl_data[:, 1]
        self.error_dl_fixed = dl_data[:, 2]  # Absolute Mpc
        
        self.da = da_data[:, 1]
        self.error_da_fixed = da_data[:, 2]  # Absolute Mpc

        self.eta_obs = self.dl / (self.da * (1 + self.z)**2)
        
        # Convert absolute fixed errors to fractional variance in eta space
        self.variance_eta_fixed = (self.eta_obs**2) * (
            (self.error_dl_fixed / self.dl)**2 + (self.error_da_fixed / self.da)**2
        )

    def get_requirements(self):
        return {'alpha': None, 'beta': None}

    def logp(self, alpha=0, beta=0, **params_values):
        
        TH = Theory(
            z=self.z, d_C=None, d_A=None, H=None, Om=None, H0=None, 
            alpha=alpha, beta=beta, xi0=1.0, model=self.EMmodel
        )
        eta_th = TH.eta

        # Propagate fractional MCMC parameters directly into eta variance
        #variance_eta_int = (self.eta_obs**2) * (sigma_uvx**2 + sigma_sarm**2)

        variance_total = self.variance_eta_fixed #+ variance_eta_int

        residuals = eta_th - self.eta_obs
        chi2_term = np.sum((residuals**2) / variance_total)
        norm_term = np.sum(np.log(2 * np.pi * variance_total))

        return -0.5 * (chi2_term + norm_term)
