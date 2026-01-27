#This module contains tools to compute the noise on d_L
#from GW observations. 
#Equations are mainly taken from ... (add arxiv numbers)


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random as r
from scipy.integrate import quad,trapz

from tools.telescope import *

gamma = 0.57722       # Euler-Mascharoni constant, used in the PN coefficients

clight  = 299792.458
kmtoMpc = 3.24078e-20


class noise:

    def __init__(self,survey):

        self.survey = survey


    def psi_func(self,M,eta,psi_0,f):
        #This function computes the psi function entering the Fourier transform of the strain.
        #Coefficients are given in 0903.0338
        #WARNING: the epression of f is different in the paper 1009.0206! What should we use?

        alpha = []

        alpha.append(1.)                                                                    #alpha_0
        alpha.append(0.)                                                                    #alpha_1
        alpha.append(3715./756+eta*55/9)                                                    #alpha_2
        alpha.append(-16*np.pi)                                                             #alpha_3
        alpha.append(15293365/508032 + eta*27145/504 + eta**2.*3085/72)                     #alpha_4
        alpha.append(np.pi*(38645/756 - eta*65/9)*(1+np.log(np.pi*M*f*6**(3./2.))))         #alpha_5
        alpha.append(11583231236531/4694215680 - np.pi**2.*640/3 - gamma*6848/21
                     + eta*(np.pi**2.*2255/12 - 15737765635/3048192) + eta**2. * 76055/1728
                     - eta**3.*127825/1296 - np.log(np.pi*M*f*64.)*6848/63)                 #alpha_6
        alpha.append(np.pi*(77096675/254016 + eta*378515/1512 - eta**2.*74045/756))         #alpha_7

        sumfunc = 0
        for ind in range(8):
#            sumfunc = sumfunc + alpha[ind]*(2*np.pi*M*f)**(ind/3)     # this is the exponent used in 1009.0206
            sumfunc = sumfunc + alpha[ind]*(2*np.pi*M*f)**((ind-5)/3) # this is the exponent used in 0903.0338

        psi = -psi_0 + (3/(256*eta))*sumfunc

        return psi


    def integrand(self,frequency,GWparameters,specs,F_plus,F_cross,total_mass,chirp_mass,eta,z,d_L):
        #This function computes the SNR integrand at a given redshift

        inclination = GWparameters['inclination'] 
        t0          = GWparameters['t0']          
        psi_0       = GWparameters['psi_0']       
        f0          = GWparameters['f0']          



        x = frequency/f0
        pi = np.pi

        # noise power spectral density S_h(f) aka horrible coefficients part one
        S = self.survey.power_spectral_densities(specs,frequency,f0)

        # phase
        Psi =  self.psi_func(total_mass,eta,psi_0,frequency/2)

        # weird varphi_(2,0) function
        varphi = np.arctan((-2*np.cos(inclination)*F_cross)/((1+np.cos(inclination)**2)*F_plus))

        # Fourier amplitude of the wave
        amplitude = (1/d_L(z))*np.sqrt((F_plus**2)*(1+np.cos(inclination)**2)**2 + 4*(F_cross**2)*np.cos(inclination)**2)*np.sqrt((5*pi)/96)*pi**(-7/6)*chirp_mass**(5/6)

        #converting all to seconds
        amplitude = amplitude*clight*kmtoMpc

        # Fourier transform of the strain H(f)
        # 1j is the python way of writing sqrt(-1)
        Fourier_strain = amplitude*frequency**(-7./6.)*np.exp(1j*(2*pi*frequency*t0 - pi/4 + 2*Psi) - varphi)

        # return the integrand in our eqn 3.7
        final = np.real((Fourier_strain*np.conj(Fourier_strain))/S) 


        return final





    def compute_noise(self,parameters,specs,GWpars,redshifts,d_L,total_mass,chirp_mass,eta):
        #This function gets survey specifications and computes the overall 
        #noise at each redshift.

        instrumental_noise = np.zeros((len(redshifts)))

        for ind,z in enumerate(redshifts):
            rho_tot = 0.

            F_plus,F_cross,f_lower,f_upper = self.survey.survey_specs(specs,GWpars,total_mass[ind])
            #Frequency sampling for integral
            freq  = np.linspace(f_lower,f_upper,100)
            dfreq = (f_upper-f_lower)/100
            for arm in range(len(F_plus)):
                #MM: check if indexing is correct. 
                intfreq = [self.integrand(f,GWpars,specs,F_plus[arm],F_cross[arm],total_mass[ind],chirp_mass[ind],eta[ind],z,d_L) for f in freq]
                rho_tot += 4*trapz(intfreq,dx=dfreq)

            instrumental_noise[ind] = 2*d_L(z)/(np.sqrt(rho_tot))

#    instrumental_noise = np.array([d_L(z)*(0.1449*z - 0.0118*z**2. + 0.0012*z**3.) for z in redshifts])

        # error due to weak lensing of the GW
        # i.e. 5% at z = 1
        # given elsewhere as 0.09z
        lens_noise = np.array([0.05*z*d_L(z) for z in redshifts])

        # total error is the square root of the sum of the squares of the instrumental and lensing errors
        noise = np.sqrt(instrumental_noise**2. + lens_noise**2.)

        return noise

