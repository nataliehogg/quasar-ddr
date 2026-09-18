import numpy as np


class QSOSpecs:
    """
    Specifications for QSO observations.

    This class provides utilities for generating observation specifications
    for QSO surveys.
    """

    # Residual dispersion of the UV--X-ray relation, in dex, per scenario.
    # 'current' is the Lusso & Risaliti scatter; 'future' and 'targeted' are
    # the improvements assumed for larger and for dedicated samples.
    UVX_RESIDUAL_DEX = {'current': 0.24, 'future': 0.17, 'targeted': 0.14}

    # Residual SARM angular-distance scatter, already a natural-log scatter.
    SARM_SIGMA_LN_DA = {'current': 0.30, 'future': 0.20, 'targeted': 0.15}

    # Slope of the UV--X-ray relation, used to convert the residual dispersion
    # into a distance dispersion
    UVX_GAMMA = 0.60

    @staticmethod
    def error_budget(scenario, gamma=None):
        """
        Natural-log distance scatters for a named error scenario.

        The UV--X-ray residual dispersion is quoted in dex and propagates to
        the luminosity distance as

            sigma_ln_dl = ln(10) * delta_uvx / (2 * (1 - gamma))

        which is Eq. (uvx_scatter_propagation) of the draft. The SARM number is
        quoted directly as a log scatter and is used as-is.

        Parameters
        ----------
        scenario : str
            One of 'current', 'future', 'targeted'.
        gamma : float, optional
            Slope of the UV--X-ray relation. Defaults to UVX_GAMMA.

        Returns
        -------
        tuple of float
            (sigma_ln_dl, sigma_ln_da)
        """
        if scenario not in QSOSpecs.UVX_RESIDUAL_DEX:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Must be one of {sorted(QSOSpecs.UVX_RESIDUAL_DEX)}"
            )
        gamma = QSOSpecs.UVX_GAMMA if gamma is None else gamma
        if gamma >= 1:
            raise ValueError("gamma must be smaller than 1")

        delta_uvx = QSOSpecs.UVX_RESIDUAL_DEX[scenario]
        sigma_ln_dl = np.log(10.0) * delta_uvx / (2.0 * (1.0 - gamma))
        sigma_ln_da = QSOSpecs.SARM_SIGMA_LN_DA[scenario]

        return sigma_ln_dl, sigma_ln_da

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
            - 'n_QSO': Number of QSO observations

        Returns
        -------
        array_like
            Array of random redshifts uniformly distributed in [z_min, z_max]

        Notes
        -----
        Currently uses uniform random distribution. This should be updated
        to use a physically motivated redshift distribution based on
        expected QSO survey selection functions.
        
        """
        
        redshifts = np.random.uniform(
            specs['z_min'],
            specs['z_max'],
            specs['n_QSO']
        )
        
        return redshifts
