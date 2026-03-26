import numpy as np
from cobaya.likelihood import Likelihood

# Import your Theory class
from tools.theory import Theory

class DDRlike(Likelihood):
    """
    Cosmology-independent Cobaya likelihood for Distance Duality Relation 
    (DDR) violation constraints. Relies on the external Theory class for 
    eta(z) formulation.
    """
    datapath: str = './mock_data/test'
    model: int = 4  # Defaulting to 4 (M1)

    def initialize(self):
        """Load mock data and construct the observational eta dataset."""
        dl_data = np.loadtxt(self.datapath + '_dl_dataset.txt')
        da_data = np.loadtxt(self.datapath + '_da_dataset.txt')

        # Assuming the mock datasets have perfectly matched redshifts
        self.z = dl_data[:, 0]
        
        self.dl = dl_data[:, 1]
        self.error_dl = dl_data[:, 2]
        
        self.da = da_data[:, 1]
        self.error_da = da_data[:, 2]

        # 1. Construct observational eta: eta_obs = d_L / (d_A * (1+z)^2)
        self.eta_obs = self.dl / (self.da * (1 + self.z)**2)

        # 2. Propagate purely STATISTICAL errors for eta_obs
        self.stat_error_eta = self.eta_obs * np.sqrt((self.error_dl / self.dl)**2 + (self.error_da / self.da)**2)

    def get_requirements(self):
        """Specify required parameters from Cobaya."""
        return {'A': None, 'B': None, 'sigma_int': None}

    def logp(self, A=0, B=0, sigma_int=0, **params_values):
        """Compute log-likelihood for given DDR parameters and intrinsic scatter."""
        
        # Determine DDR model type (excluding cosmology-dependent ones)
        model_map = {3: 'epspar', 4: 'M1', 5: 'M2', 6: 'M3'}
        EMmodel = model_map.get(self.model, 'M1')

        # ---------------------------------------------------------
        # THEORETICAL ETA (Via Theory class)
        # ---------------------------------------------------------
        # We pass None/dummy variables for the cosmology parameters 
        # (d_C, d_A, H, Om, H0) because phenomenological models don't use them.
        TH = Theory(
            z=self.z, 
            d_C=None, 
            d_A=None, 
            H=None, 
            Om=None, 
            H0=None, 
            A=A, 
            B=B, 
            xi0=1.0, 
            model=EMmodel
        )
        eta_th = TH.eta

        # ---------------------------------------------------------
        # INTRINSIC SCATTER & VARIANCE
        # ---------------------------------------------------------
        # A fractional error on d_L translates to a fractional error on eta.
        # So absolute scatter on eta = eta_obs * sigma_int
        variance_eta = self.stat_error_eta**2 + (self.eta_obs * sigma_int)**2

        # ---------------------------------------------------------
        # TOTAL LOG-LIKELIHOOD
        # ---------------------------------------------------------
        residuals = eta_th - self.eta_obs
        
        chi2_term = np.sum((residuals**2) / variance_eta)
        norm_term = np.sum(np.log(2 * np.pi * variance_eta))

        return -0.5 * (chi2_term + norm_term)
