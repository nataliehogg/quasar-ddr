#This module contains functions to compute masses of the binaries
#and their distributions in redshift and on the sky

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random as r
from scipy.integrate import quad,trapezoid

conv_fac=4.93*10**(-6.) #solar masses to seconds (who measures masses in seconds???)



class GWdist:

    def __init__(self,d_C,H):

        self.d_C = d_C
        self.H   = H

    def merger_rate(self,model,zs):
        #This function returns the merger rate as a function of redshift.
        #Having it separate might make it easier to modify the model later

        if model=='standard':
           rate = [(1+(2*z)) if z <=1 else (3/4)*(5-z) if 1 < z < 5 else 0 for z in zs]
        else:
           print('ERROR: the only models I know are "{}"'.format(model))
           exit()

        return rate


    def redshift_distribution(self,specs,GWpars):

        d_C = self.d_C
        H   = self.H
        #  creating the custom distribution for d_L(z)
        sampling = np.linspace(specs['z_min'], specs['z_max'], 1000)

        distribution = self.probability(specs,GWpars)

        normed = distribution(sampling)

        redshifts = r.choices(sampling, normed, k = specs['GW_number']) # draw from our custom normalised distribution

        # redshifts is now the list of redshifts for which we get mock luminosity distances and their errors

        return np.sort(redshifts)

    def probability(self,specs,GWpars):

        d_C = self.d_C
        H   = self.H

        z_range = np.linspace(specs['z_min'],specs['z_max'],1000)
        dz = (specs['z_max']-specs['z_min'])/1000


        if GWpars['merger_distribution'] == 'weighted':
           burst_rate = self.merger_rate(GWpars['merger_model'],z_range)
           unnorm   = (4*np.pi*(d_C(z_range)**2)*burst_rate)/(H(z_range)*(1+z_range))
           integral = trapezoid(unnorm,dx=dz) 
           norm     = specs['GW_number']/integral
           finaldist= norm*unnorm

        if GWpars['merger_distribution'] == 'uniform':
           uniform  = (4*np.pi*(d_C(z_range)**2))/(H(z_range)*(1+z_range)) 
           integral = trapezoid(uniform,dx=dz)
           norm     = specs['GW_number']/integral
           finaldist= norm*uniform

        distribution = interp1d(z_range,finaldist,kind='cubic')

        return distribution 


    def masses(self,merger_type,redshifts):
        #This function return the masses of the two binaries

        if merger_type == 'NS-NS':
           NSmass = 1.4 *conv_fac
           mass_1 = np.array([(1+z)*NSmass for z in redshifts])
           mass_2 = np.array([(1+z)*NSmass for z in redshifts])
        else:
           print('ERROR: mock maker only working for NS-NS mergers')
           exit()

        total_mass = mass_1 + mass_2               # total mass now a np.array
        eta        = (mass_1*mass_2)/total_mass**2 # symmetric mass ratio
        chirp_mass = total_mass*(eta**(3/5))       # chirp mass in solar masse

        return total_mass,chirp_mass,eta

    def sky_distribution(self,GWpars):

        # sky positions in radians
        if GWpars['sky_distribution'] == 'uniform':
           random_1, random_2 = np.random.random(2).reshape(2, -1)
           theta = 2.*np.pi*(random_1 - 0.5)
           phi = np.arcsin(2.*(random_2 -0.5))

        else:
           print('ERROR: unknown assumption for sky distribution')
           exit()

        return theta[0],phi[0]
