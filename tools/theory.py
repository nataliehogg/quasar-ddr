from scipy.interpolate import interp1d

from .constants import SPEED_OF_LIGHT_KMS


class Theory:
    """
    Compute modified cosmological distances with DDR violations.

    This class implements three different models for Distance Duality Relation
    (DDR) violations in the electromagnetic sector:
     - M1: constant DDR violation, eta = 1 + alpha
     - M2: linear DDR violation, eta = 1 + alpha + beta*z
     - M3: power law DDR violation, eta = (1+z)**alpha

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
    alpha : float
        DDR violation amplitude parameter
    beta : float
        DDR violation linear term parameter
    model : str
        Model type: 'M1', 'M2', 'M3'

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

    def __init__(self, z, d_C, d_A, H, Om, H0, alpha, beta, model):

        # Validate model type
        valid_models = ['M1', 'M2', 'M3']
        if model not in valid_models:
            raise ValueError(
                f"Invalid model type '{model}'. "
                f"Must be one of {valid_models}"
            )

        self.z = z
        self.d_A = d_A
        self.H0 = H0
	
	
        # Compute DDR violation factor eta(z) based on model
        if model == 'M1':
            # Model 1: constant DDR violation 
            self.eta = 1 + alpha
        elif model == 'M2':
            # Model 2: linear DDR violation
            self.eta = 1 + alpha + beta*z
        elif model == 'M3':
            # Model 3: power-law DDR violation
            self.eta = (1+z)**alpha 
        
        """
        If we let the z of transition to be a free parameter of the model we could also use these models
        elif model == 'M1*':
            # Model 1: constant DDR violation with transition at z*
            if (z < 1090):
            	self.eta = 1 + A
            else: 
            	self.eta = 1 + B
        elif model == 'M3*'
            # Model 3: power-law DDR violation with transition at z*
            if (z < 1090):
            	self.eta = (1+z)**A
            else: 
                self.eta = (1+z)**B  
        """
         

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


