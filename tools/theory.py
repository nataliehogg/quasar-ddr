import numpy as np
from scipy.interpolate import interp1d

from .constants import SPEED_OF_LIGHT_KMS


class Theory:
    """
    Compute modified cosmological distances with DDR violations.

    This class implements three different models for Distance Duality Relation
    (DDR) violations in the electromagnetic sector:
    - 'coherent': Violation depends on Hubble parameter deviation
    - 'incoherent': Violation depends on comoving distance
    - 'epspar': Power-law violation parameterization

    Parameters
    ----------
    z : array_like
        Redshift array for distance calculations
    d_C : callable
        Comoving distance interpolation function d_C(z)
    d_A : callable
        Angular diameter distance interpolation function d_A(z)
    H : callable
        Hubble parameter interpolation function H(z)
    Om : float
        Matter density parameter Omega_m
    H0 : float
        Hubble constant in km/s/Mpc
    A : float
        DDR violation amplitude parameter
    xi0 : float
        DDR violation transition scale parameter
    model : str
        Model type: 'coherent', 'incoherent', or 'epspar'

    Attributes
    ----------
    z : array_like
        Redshift array
    d_A : callable
        Angular diameter distance function
    H0 : float
        Hubble constant
    eta : array_like
        DDR violation factor eta(z)
    """

    def __init__(self, z, d_C, d_A, H, Om, H0, A, xi0, model):

        # Validate model type
        valid_models = ['coherent', 'incoherent', 'epspar']
        if model not in valid_models:
            raise ValueError(
                f"Invalid model type '{model}'. "
                f"Must be one of {valid_models}"
            )

        self.z = z
        self.d_A = d_A
        self.H0 = H0

        # Compute DDR violation factor eta(z) based on model
        if model == 'coherent':
            # Coherent model: violation depends on Hubble parameter deviation
            P = A + (1 - A) * np.exp(-(H(z) - H0) / (Om * H0) * xi0)
            self.eta = np.sqrt(1 / P)
        elif model == 'incoherent':
            # Incoherent model: violation depends on comoving distance
            P = A + (1 - A) * np.exp((-3/2) * (H0 / SPEED_OF_LIGHT_KMS) * d_C(z) * xi0)
            self.eta = np.sqrt(1 / P)
        elif model == 'epspar':
            # Epsilon parameterization: power-law violation
            self.eta = (1 + z)**A          

    def EMlumdist(self):
        """
        Compute modified electromagnetic luminosity distance.

        Returns an interpolation function for the luminosity distance
        in the electromagnetic sector, including DDR violation effects.

        Returns
        -------
        callable
            Interpolation function d_L^EM(z) in Mpc

        Notes
        -----
        The modified luminosity distance is:
        d_L^EM(z) = (H0/c) * (1+z)^2 * d_A(z) * eta(z)

        where eta(z) encodes the DDR violation.
        """
        lumdist = self.H0 / SPEED_OF_LIGHT_KMS * (1 + self.z)**2 * self.d_A(self.z) * self.eta

        return interp1d(self.z, lumdist, kind='cubic')


