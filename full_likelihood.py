import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import trapz

from tools.theory import *

debug = False


datapath = './mock_data/LCDM'

GWpath   = datapath+'_GWdataset.txt'
SNpath   = datapath+'_SNdataset.txt'
BAOpath  = datapath+'_BAOdataset.txt'

#reading data and creating dataset here------------
GWdata = np.loadtxt(GWpath)
zgw    = GWdata[:,0]
dLgw   = GWdata[:,1]
errgw  = GWdata[:,2]

SNdata = np.loadtxt(SNpath)
zsn    = SNdata[:,0]
musn   = SNdata[:,1]
errsn  = SNdata[:,2]
invSNcovmat = np.zeros((len(zsn),len(zsn)))
for ind in range(len(zsn)):
    invSNcovmat[ind,ind] = 1/errsn[ind]**2.

BAOdata = np.loadtxt(BAOpath)
zbao    = BAOdata[:,0]
dabao   = BAOdata[:,1]
errda   = BAOdata[:,2]
Hbao    = BAOdata[:,3]
errH    = BAOdata[:,4]

z_range = np.linspace(0.001,5.,1000)

clight = 299792.458


def DDRlike(omegam=0.314,A=1.,xi0=1.,Ga=0.,Gn=0.,M0=-19.5,model=1,use_GW=True,use_SN=True,use_BAO=True,
            _theory={"angular_diameter_distance": {"z": z_range}, "H": {"z": z_range}}):

    #STANDARD COSMOLOGY
    if model == 1: EMmodel = 'incoherent'
    if model == 2: EMmodel = 'coherent'
    if model == 3: EMmodel = 'epspar'

    H0 = _theory.get_param('H0')

    z       = np.linspace(0.,5.,1000)

    Ez         = np.sqrt(omegam*(1+z)**3+(1-omegam))
    hubble     = interp1d(z,H0*Ez,kind='cubic')

    com = []
    for zi in z_range:
        zint  = np.linspace(0,zi,100)
        dz    = (zi-0.)/100
        com.append(clight*trapz(1/hubble(zint),zint,dx=dz))

    comoving        = interp1d(z_range,com,kind='cubic')
    dA              = interp1d(z_range,com/(1+z),kind='cubic')
    hubble_mpc      = interp1d(z,hubble(z)/clight,kind='cubic')

    lumdist = interp1d(z_range,(1+z)*com,kind='cubic')


    #Importing theory stuff!
    #This computes modifications
    TH = theory(z_range,comoving,dA,hubble,omegam,_theory.get_param('H0'),A,xi0,Ga,Gn,M0,EMmodel)

    lumdist_GW = TH.GWobs()
    mu_EM = TH.EMobs()

    chi2GW = 0
    chi2SN = 0
    chi2dA = 0
    chi2H  = 0


    if use_GW:
       for ind,z in enumerate(zgw):
           chi2GW += ((dLgw[ind]-lumdist_GW(z))/errgw[ind])**2.

       if debug:
           print('chi2GW = {}'.format(chi2GW))
           plt.figure()
           plt.title('GW luminosity distance')
           plt.plot(z_range,lumdist_GW(z_range),color='red',label='Fiducial')
           plt.errorbar(zgw,dLgw,yerr=errgw,marker='*',ls='',label='Einstein Telescope')
           plt.xlabel(r'$z$')
           plt.ylabel(r'$d_L(z)$')
           plt.legend(loc='upper left')
           plt.show()

    if use_BAO:
       for ind,z in enumerate(zbao):
           chi2dA += ((dabao[ind]-dA(z))/errda[ind])**2.
           chi2H  += ((Hbao[ind]-hubble(z))/errH[ind])**2.

       if debug:
           print('chi2BAO (dA) = {}'.format(chi2dA))
           print('chi2BAO (H)  = {}'.format(chi2H))
           plt.figure()
           plt.title('BAO angular distance')
           plt.plot(z_range,dA(z_range),color='red',label='Fiducial')
           plt.errorbar(zbao,dabao,yerr=errda,marker='*',ls='',label='DESI')
           plt.xlabel(r'$z$')
           plt.ylabel(r'$d_A(z)$')
           plt.legend(loc='lower right')
           plt.show()

           plt.figure()
           plt.title('BAO Hubble parameter')
           plt.plot(z_range,hubble(z_range),color='red',label='Fiducial')
           plt.errorbar(zbao,Hbao,yerr=errH,marker='*',ls='',label='DESI')
           plt.xlabel(r'$z$')
           plt.ylabel(r'$H(z)$')
           plt.legend(loc='upper left')
           plt.show()



    if use_SN:
       SNdiff  = np.array(musn-mu_EM(zsn))
       idt     = [1.]*len(SNdiff)

       a       = np.dot(SNdiff,np.dot(invSNcovmat,SNdiff))
       b       = np.dot(SNdiff,np.dot(invSNcovmat,idt))
       e       = np.dot(idt,np.dot(invSNcovmat,idt))

       chi2SN  = a+np.log10(e/(2*np.pi))-b**2./e
       if debug:
           print('chi2SN = {}'.format(chi2SN))
           plt.figure()
           plt.title('SN distance modulus')
           plt.plot(z_range,mu_EM(z_range)+25-5*np.log10(_theory.get_param('H0'))-9,color='red',label='Fiducial')
           plt.errorbar(zsn,musn,yerr=errsn,marker='*',ls='',label='LSST')
           plt.xlabel(r'$z$')
           plt.ylabel(r'$\mu(z)$')
           plt.legend(loc='lower right')
           plt.show()



    chi2 = chi2GW + chi2SN + chi2dA + chi2H

    return -chi2/2
