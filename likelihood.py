import numpy as np
from cobaya.likelihood import Likelihood

from tools.theory import Theory
from tools.utilities import Utilities

class DDRlike(Likelihood):
    """
    Cosmology-independent Cobaya likelihood for Distance Duality Relation
    (DDR) violation constraints using alpha and beta parameters.

    The comparison is made in ln eta rather than eta. Forming eta = d_L /
    (d_A (1+z)^2) and propagating its error as
    eta * sqrt((sigma_dl/d_L)^2 + (sigma_da/d_A)^2) puts the measured distances
    in the denominators, so each object's weight depends on its own noise:
    objects whose d_A scatters high get up-weighted exactly where their eta is
    low. That biases alpha low by an amount which does not shrink with sample
    size. In log space the two distances enter additively and the bias goes.
    """
    datapath: str = './mock_data/test'
    model: str = 'M1'  # DDR violation model: 'M1', 'M2' or 'M3'

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

        # Observational ln eta and its variance. The datasets store absolute
        # errors in Mpc; sigma/d is the natural-log scatter. The variance is
        # fixed by the data, so the Gaussian normalisation is a constant and
        # is dropped below.
        self.log_eta_obs = np.log(self.dl) - np.log(self.da) - 2*np.log1p(self.z)
        self.var_log_eta = (self.error_dl/self.dl)**2 + (self.error_da/self.da)**2

    def get_requirements(self):
        """Require only the parameters this model actually samples.

        Uses the same MODEL_PARAMS mapping as Utilities.sampler_config, so the
        required parameters always match those declared in the config
        (M1/M3: alpha; M2: alpha, beta).
        """
        return {name: None for name in Utilities.MODEL_PARAMS[self.model]}

    def logp(self, alpha=0, beta=0, **params_values):
        """Compute log-likelihood for DDR parameters alpha/beta."""

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

        # Reject unphysical eta rather than taking the log of a negative number
        if np.any(eta_th <= 0):
            return -np.inf

        chi2 = np.sum((np.log(eta_th) - self.log_eta_obs)**2 / self.var_log_eta)

        return -chi2/2
