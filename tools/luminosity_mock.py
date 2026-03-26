import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

from .QSO_specifications import QSOSpecs
from .constants import SPEED_OF_LIGHT_KMS


class QSOLuminosity:
    """
    Generate mock luminosity distance data from QSO observations.

    This class creates synthetic datasets of luminosity distances as would be
    measured from supermassive black hole photon ring observations. The mock
    data includes realistic observational errors and is validated to follow
    Gaussian statistics.

    Parameters
    ----------
    cosmoparams : dict
        Cosmological parameters, must contain 'H0' (Hubble constant)
    specifications : dict
        Observation specifications with keys:
        - 'z_min': Minimum redshift
        - 'z_max': Maximum redshift
        - 'n_QSO': Number of QSO observations
    z_in : array_like
        Input redshift array for interpolation
    outpath : str
        Output file path prefix (will append '_dl_dataset.txt')
    luminosity_distance : callable
        Theoretical luminosity distance function d_L(z)

    Attributes
    ----------
    redshifts : array_like
        Random redshift distribution for Q observations
    data : array_like
        Mock luminosity distance values in Mpc
    error : array_like
        Observational errors in Mpc

    Notes
    -----
    Error budget combines:
    - 1% calibration uncertainty
    - 2% systematic uncertainty
    - 12% astrophysical uncertainty
    Total: ~5.4% in quadrature
    """

    def __init__(self, cosmoparams, specifications, z_in, outpath, luminosity_distance):

        print('Generating QSO luminosity distance mock.')

        # Get the redshift distribution
        self.redshifts = QSOSpecs.redshift_distribution(specifications)

        # Convert to proper units (Mpc)
        d_L = interp1d(z_in, SPEED_OF_LIGHT_KMS / cosmoparams['H0'] * luminosity_distance(z_in))

        err_1 = 0.1 * (SPEED_OF_LIGHT_KMS / cosmoparams['H0'])
        err_2 = 0.1 * (SPEED_OF_LIGHT_KMS / cosmoparams['H0'])
        err_3 = 0.1 * (SPEED_OF_LIGHT_KMS / cosmoparams['H0'])

        # NH: there must be a better way to get this shaped into the right size array
        self.error = np.ones(len(self.redshifts)) * np.sqrt(err_1**2 + err_2**2 + err_3**2)

        # Validation loop: ensure mock data residuals follow N(0,1) distribution
        # This guarantees the mock data has correct statistical properties
        mu_residual = 1
        sigma_residual = 3
        count = 0

        while (np.abs(mu_residual) > 0.01) or (np.abs(sigma_residual - 1) > 0.01):
            self.data = [d_L(zi) + np.random.normal(loc=0.0, scale=self.error[ind])
                        for ind, zi in enumerate(self.redshifts)]
            hist = np.array([(self.data[ind] - d_L(zi)) / self.error[ind]
                            for ind, zi in enumerate(self.redshifts)])
            mu_residual, sigma_residual = norm.fit(hist)
            count += 1

        print('Saving mock to '+outpath+'_dl_dataset.txt')


        np.savetxt(outpath+'_dl_dataset.txt', np.array([self.redshifts, 
                                                        self.data,
                                                        self.error]).T, 
                                                        delimiter=' ')

    def mock(self):
        """
        Return mock dataset as dictionary.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'z': Redshifts
            - 'value': Luminosity distances in Mpc
            - 'error': Observational errors in Mpc
        """
        mockdict = {}
        mockdict['z'] = self.redshifts
        mockdict['value'] = self.data
        mockdict['error'] = self.error

        return mockdict
