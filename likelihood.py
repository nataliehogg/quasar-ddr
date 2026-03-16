import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from cobaya.likelihood import Likelihood

from tools.theory import Theory
from tools.constants import SPEED_OF_LIGHT_KMS

# this is the change for the quasar branch


class DDRlike(Likelihood):
    """
    Cobaya likelihood for Distance Duality Relation (DDR) violation constraints.

    Uses mock SMBH shadow observations to constrain cosmological parameters
    and potential DDR violations in the electromagnetic sector.

    Parameters
    ----------
    datapath : str, optional
        Path prefix to mock data files (default: './mock_data/test')
    model : int, optional
        DDR violation model: 1=incoherent, 2=coherent, 3=epspar (default: 1)
    """

    # Class attributes that can be overridden in Cobaya input
    datapath: str = './mock_data/test'
    model: int = 1

    # Redshift range for distance calculations
    z_range = np.linspace(0.001, 5., 1000)

    def initialize(self):
        """Load mock data and pre-compute integration grid."""
        # Load mock datasets
        dl_data = np.loadtxt(self.datapath + '_dl_dataset.txt')
        da_data = np.loadtxt(self.datapath + '_da_dataset.txt')

        self.z_dl = dl_data[:, 0]
        self.data_dl = dl_data[:, 1]
        self.error_dl = dl_data[:, 2]

        self.z_da = da_data[:, 0]
        self.data_da = da_data[:, 1]
        self.error_da = da_data[:, 2]

        # Pre-compute integration structure for comoving distance
        self.n_int_points = 100
        self._setup_integration_grid()

    def _setup_integration_grid(self):
        """Pre-compute integration grid structure for comoving distance calculation."""
        self.z_int_grids = []
        self.dz_values = []

        for zi in self.z_range:
            # Start integration from small epsilon to avoid interpolation issues at z=0
            z_int = np.linspace(0, zi, self.n_int_points)
            dz = (zi - 0.) / self.n_int_points
            self.z_int_grids.append(z_int)
            self.dz_values.append(dz)

    def get_requirements(self):
        """Specify required parameters from Cobaya."""
        return {'H0': None, 'A': None, 'Omegam': None}

    def logp(self, A=1, Omegam=0.314, **params_values):
        """
        Compute log-likelihood for given parameters.

        Parameters
        ----------
        A : float
            DDR violation amplitude parameter
        Omegam : float
            Matter density parameter
        **params_values : dict
            Additional parameters (unused but required by Cobaya)

        Returns
        -------
        float
            Log-likelihood value (-chi^2/2)
        """
        # Fixed parameters
        xi0 = 1.0  # DDR violation transition scale (fixed)

        # Determine DDR model type
        model_map = {1: 'incoherent', 2: 'coherent', 3: 'epspar'}
        EMmodel = model_map.get(self.model, 'incoherent')

        # Get H0 from Cobaya provider
        H0 = self.provider.get_param('H0')

        # Compute Hubble parameter as function of redshift
        # Use extended range from 0 to handle integration properly
        z_extended = np.linspace(0., 5., 1000)
        hubble_values = H0 * np.sqrt(Omegam * (1 + z_extended)**3 + (1 - Omegam))
        hubble = interp1d(z_extended, hubble_values, kind='cubic')

        # Compute comoving distance using pre-computed integration grids
        com = []
        for z_int, dz in zip(self.z_int_grids, self.dz_values):
            com.append(SPEED_OF_LIGHT_KMS * trapezoid(1/hubble(z_int), z_int, dx=dz))

        # Create distance interpolation functions
        comoving = interp1d(self.z_range, com, kind='cubic')
        dA = interp1d(self.z_range, com / (1 + self.z_range), kind='cubic')

        # Compute modified luminosity distance with DDR violation
        TH = Theory(self.z_range, comoving, dA, hubble, Omegam, H0, A, xi0, EMmodel)
        luminosity_EM = TH.EMlumdist()

        # Convert theoretical prediction to Mpc (to match mock data units)
        # Mock generation does: d_L = (c/H0) * luminosity_distance(z)
        # So we must apply the same conversion here
        conversion_factor = SPEED_OF_LIGHT_KMS / H0  # Mpc

        # Compute chi-squared for luminosity distance
        residuals_dL = (self.data_dl - conversion_factor * luminosity_EM(self.z_dl)) / self.error_dl
        chi2dL = np.sum(residuals_dL**2)

        # Compute chi-squared for angular diameter distance
        residuals_dA = (self.data_da - dA(self.z_da)) / self.error_da
        chi2dA = np.sum(residuals_dA**2)

        # Total chi-squared
        chi2 = chi2dL + chi2dA

        return -chi2 / 2
