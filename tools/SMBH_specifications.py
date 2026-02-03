import numpy as np


class SMBHSpecs:
    """
    Specifications for SMBH shadow observations.

    This class provides utilities for generating observation specifications
    for supermassive black hole shadow/photon ring surveys.
    """

    @staticmethod
    def redshift_distribution(specs):
        """
        Generate redshift distribution for SMBH observations.

        Parameters
        ----------
        specs : dict
            Specifications dictionary with keys:
            - 'z_min': Minimum redshift
            - 'z_max': Maximum redshift
            - 'n_SMBH': Number of SMBH observations

        Returns
        -------
        array_like
            Array of random redshifts uniformly distributed in [z_min, z_max]

        Notes
        -----
        Currently uses uniform random distribution. This should be updated
        to use a physically motivated redshift distribution based on
        expected SMBH survey selection functions.
        """
        redshifts = np.random.uniform(
            specs['z_min'],
            specs['z_max'],
            specs['n_SMBH']
        )

        return redshifts
