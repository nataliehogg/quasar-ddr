import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d


clight = 299792.458

class theory:

    def __init__(self,z,d_C,d_A,H,Om,H0,A,xi0,Ga,Gn,M0,model):


        self.z   = z
        self.d_A = d_A
        self.H0  = H0
        self.M0  = M0


        self.Gratio = 1+Ga*(z/(1+z))**Gn-Ga*(z/(1+z))**(2*Gn) #This is Geff/G

        if model == 'coherent':
           P = A+(1-A)*np.exp(-(H(z)-H0)/(Om*H0)*xi0) #Pcoherent
           self.eta = np.sqrt(1/P)
        elif model == 'incoherent':
           P = A+(1-A)*np.exp((-3/2)*(H0/clight)*d_C(z)*xi0) #Pincoherent
           self.eta = np.sqrt(1/P)
        elif model == 'epspar':
           self.eta = (1+z)**A
           



    def GWobs(self):


        #This is Eq.(2.17) of 1906.01593
        #WARNING! It assumes d_EM = (1+z)**2. d_A
        lumdist = np.sqrt(self.Gratio)*(1+self.z)**2.*self.d_A(self.z)

        return interp1d(self.z, lumdist, kind='cubic')

    def EMlumdist(self):

        lumdist = self.H0/clight*(1+self.z)**2.*self.d_A(self.z)*self.eta

        return interp1d(self.z, lumdist, kind='cubic')

    def EMobs(self,absolute=True):

        lumdist = self.EMlumdist()

        if absolute:
           mu   = 5*np.log10(lumdist(self.z))+25-(15/4)*np.log10(self.Gratio)
        else:
           mu   = 5*np.log10(lumdist(self.z))+25-5*np.log10(self.H0/clight)+self.M0-(15/4)*np.log10(self.Gratio)

        return interp1d(self.z, mu, kind='cubic')
