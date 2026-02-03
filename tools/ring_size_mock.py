import numpy as np
from scipy.stats import norm

from .SMBH_specifications import SMBHSpecs
from .constants import SPEED_OF_LIGHT_KMS


class RingSize:
    """
    Generate mock angular diameter distance data from SMBH shadow observations.

    This class creates synthetic datasets of angular diameter distances as
    would be measured from supermassive black hole shadow size observations.
    The mock data includes realistic observational errors and is validated
    to follow Gaussian statistics.

    Parameters
    ----------
    cosmoparams : dict
        Cosmological parameters, must contain 'H0' (Hubble constant)
    specifications : dict
        Observation specifications with keys:
        - 'z_min': Minimum redshift
        - 'z_max': Maximum redshift
        - 'n_SMBH': Number of SMBH observations
    z_in : array_like
        Input redshift array for interpolation
    outpath : str
        Output file path prefix (will append '_da_dataset.txt')
    angular_distance : callable
        Theoretical angular diameter distance function d_A(z)

    Attributes
    ----------
    redshifts : array_like
        Random redshift distribution for SMBH observations
    data : array_like
        Mock angular diameter distance values in Mpc
    error : array_like
        Observational errors in Mpc

    Notes
    -----
    Error budget: ~1% systematic uncertainty
    """

    def __init__(self, cosmoparams, specifications, z_in, outpath, angular_distance):

        print('Generating SMBH angular distance mock.')

        # Get the redshift distribution
        self.redshifts = SMBHSpecs.redshift_distribution(specifications)

        # Error budget (in Mpc)
        err_1 = 0.01 * (SPEED_OF_LIGHT_KMS / cosmoparams['H0'])

        # NH: as in the dL mock, this needs work
        self.error = np.ones(len(self.redshifts)) * np.sqrt(err_1**2)

        # Validation loop: ensure mock data residuals follow N(0,1) distribution
        # This guarantees the mock data has correct statistical properties
        mu_residual = 1
        sigma_residual = 3
        count = 0

        while (np.abs(mu_residual) > 0.01) or (np.abs(sigma_residual - 1) > 0.01):
            self.data = [angular_distance(zi) + np.random.normal(loc=0.0, scale=self.error[ind])
                        for ind, zi in enumerate(self.redshifts)]
            hist = np.array([(self.data[ind] - angular_distance(zi)) / self.error[ind]
                            for ind, zi in enumerate(self.redshifts)])
            mu_residual, sigma_residual = norm.fit(hist)
            count += 1

        print('Saving mock to '+outpath+'_da_dataset.txt')

        np.savetxt(outpath+'_da_dataset.txt', np.array([self.redshifts, 
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
            - 'value': Angular diameter distances in Mpc
            - 'error': Observational errors in Mpc
        """
        mockdict = {}
        mockdict['z'] = self.redshifts
        mockdict['value'] = self.data
        mockdict['error'] = self.error

        return mockdict