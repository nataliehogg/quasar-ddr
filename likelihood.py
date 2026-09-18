import numpy as np
from cobaya.likelihood import Likelihood

from tools.qso_mock import load_paired_log_mock
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
    low. That biases alpha by an amount which does not shrink with sample size.
    In log space the two distances enter additively and the bias goes.

    For the same reason the log scatters are read straight from the paired
    dataset rather than rebuilt from an absolute error column. Dividing a stored
    sigma in Mpc by a measured distance puts that object's own noise back into
    its own weight, which reintroduces the same bias.
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

        mock = load_paired_log_mock(self.datapath + '_paired_dataset.npz')
        self.z = mock.redshift

        # This likelihood treats the two distances as independent
        if np.any(mock.rho != 0.0):
            raise ValueError(
                "dataset was generated with rho != 0, but this likelihood "
                "assumes independent d_L and d_A errors"
            )

        # Observational ln eta and its variance. The log scatters are a property
        # of the survey, supplied with the data, so the variance does not depend
        # on the sampled parameters and the Gaussian normalisation is a constant
        # which is dropped below.
        self.log_eta_obs = (np.log(mock.dl_observed) - np.log(mock.da_observed)
                            - 2*np.log1p(mock.redshift))
        self.var_log_eta = mock.sigma_ln_dl**2 + mock.sigma_ln_da**2

    def get_requirements(self):
        """Require only the parameters this model actually samples.

        Uses the same MODEL_PARAMS mapping as Utilities.sampler_config, so the
        required parameters always match those declared in the config
        (M1/M3: alpha; M2: alpha, beta).
        """
        return {name: None for name in Utilities.MODEL_PARAMS[self.model]}

    def logp(self, alpha=0, beta=0, **params_values):
        """Compute log-likelihood for DDR parameters alpha/beta."""

        # Calculate theoretical eta using updated nomenclature. Theory rejects
        # a non-positive eta(z) by raising, but for a sampler that is just the
        # edge of the prior, so it is reported as zero likelihood rather than
        # allowed to abort the chain.
        try:
            TH = Theory(
                z=self.z,
                d_C=None,
                d_A=None,
                H=None,
                Om=None,
                H0=None,
                alpha=alpha,
                beta=beta,
                xi0=1.0,
                model=self.EMmodel
            )
        except ValueError:
            return -np.inf
        eta_th = TH.eta

        chi2 = np.sum((np.log(eta_th) - self.log_eta_obs)**2 / self.var_log_eta)

        return -chi2/2
