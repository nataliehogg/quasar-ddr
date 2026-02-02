import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from cobaya.likelihood import Likelihood

from tools.theory import Theory


z_range = np.linspace(0.001,5.,1000)

clight = 299792.458


class DDRlike(Likelihood):

    def initialise(self):
        datapath = './mock_data/test'
        dl_data = np.loadtxt(datapath+'_dl_dataset.txt')
        da_data = np.loadtxt(datapath+'_da_dataset.txt')

        self.z_dl   = dl_data[:,0]
        self.data_dl = dl_data[:,1]
        self.error_dl = dl_data[:,2]

        self.z_da    = da_data[:,0]
        self.data_da   = da_data[:,1]
        self.error_da   = da_data[:,2]

    def get_requirements(self):

        return {'H0': None, 'A': None, 'xi0': None} # NH: omega_m?

    def logp(self, A=1, xi0=1, **params_values):

        omegam=0.3 # NH: watch out

        model = 1

        A = params_values['A']
        xi0 = params_values['xi0']

        #STANDARD COSMOLOGY
        if model == 1:
            EMmodel = 'incoherent'
        if model == 2:
            EMmodel = 'coherent'
        if model == 3:
            EMmodel = 'epspar'

        H0 = self.provider.get_param('H0')

        z = np.linspace(0.,5.,1000)

        hubble = interp1d(z, H0*np.sqrt(omegam*(1+z)**3+(1-omegam)),kind='cubic')

        com = []
        for zi in z_range:
            zint  = np.linspace(0,zi,100)
            dz    = (zi-0.)/100
            com.append(clight*trapezoid(1/hubble(zint),zint,dx=dz))

        comoving        = interp1d(z_range,com,kind='cubic')
        dA              = interp1d(z_range,com/(1+z),kind='cubic')

        #Importing theory stuff!
        #This computes modifications
        TH = Theory(z_range, comoving, dA, hubble, omegam, H0, A, xi0, EMmodel)

        luminosity_EM = TH.EMlumdist()

        chi2dL = 0
        chi2dA = 0

        for ind,z in enumerate(self.z_dl):
            chi2dL += ((self.data_dl[ind]-luminosity_EM(z))/self.error_dl[ind])**2.

        for ind,z in enumerate(self.z_da):
            chi2dA += ((self.data_da[ind]-dA(z))/self.error_da[ind])**2.

        chi2 = chi2dL + chi2dA

        return -chi2/2
