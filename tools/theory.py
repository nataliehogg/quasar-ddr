import numpy as np
from scipy.interpolate import interp1d

clight = 299792.458

class Theory:

    def __init__(self, z, d_C, d_A, H, Om, H0, A, xi0, model):


        self.z   = z
        self.d_A = d_A
        self.H0  = H0

        if model == 'coherent':
           P = A+(1-A)*np.exp(-(H(z)-H0)/(Om*H0)*xi0) #Pcoherent
           self.eta = np.sqrt(1/P)
        elif model == 'incoherent':
           P = A+(1-A)*np.exp((-3/2)*(H0/clight)*d_C(z)*xi0) #Pincoherent
           self.eta = np.sqrt(1/P)
        elif model == 'epspar':
           self.eta = (1+z)**A          

    def EMlumdist(self, eta):

        lumdist = self.H0/clight*(1+self.z)**2.*self.d_A(self.z)*self.eta

        return interp1d(self.z, lumdist, kind='cubic')


