import numpy as np
from cobaya.likelihood import Likelihood

# Import your Theory class
from tools.theory import Theory

class DDRlike(Likelihood):
    """
    Cosmology-independent Cobaya likelihood for Distance Duality Relation 
    (DDR) violation constraints using alpha and beta parameters.
    """
    datapath: str = './mock_data/test'
    model: int = 4  # Defaulting to 4 (M1)

    def initialize(self):
        """Load mock data and identify model selection once at startup."""
        
        # Identify model selection for the Theory class
        model_map = {1: 'M1', 2: 'M2', 3: 'M3'}
        self.EMmodel = model_map.get(self.model, 'M1')
        
        # This print statement will only appear once when Cobaya starts
        print(f"--- [DDRlike] Model Selected: {self.EMmodel} (ID: {self.model}) ---")
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
        """Specify alpha and beta as required parameters from Cobaya."""
        return {'alpha': None, 'beta': None, 'sigma_int': None}

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
