import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random as r
from scipy.integrate import quad



class survey:

    def __init__(self,GWdist):

        self.GWdist = GWdist

    def antenna_pattern(self,theta,phi,pol):
        #This function returns the single antenna pattern  given the angles.
        # note that this will need to be changed if a different telescope is used

        F_plus = (np.sqrt(3)/2)*((1/2)*(1+(np.cos(theta)**2)) * np.cos(2*phi)*(np.cos(2*pol)) - np.cos(theta)*np.sin(2*phi)*np.sin(2*pol))
        F_cross = (np.sqrt(3)/2)*((1/2)*(1+(np.cos(theta)**2)) * np.cos(2*phi)*(np.sin(2*pol)) + np.cos(theta)*np.sin(2*phi)*np.cos(2*pol))

        return F_plus,F_cross

    def power_spectral_densities(self,specs,f,f0):
        # Taken from 0903.0338

        x = f/f0

        if specs['survey'] == 'Einstein Telescope':

            S0 = 1.5*10**(-52.)

            alpha = -4.1
            beta  = -0.69
            a0    = 186
            b0    = 233
            b1    = 31
            b2    = -65
            b3    = 52
            b4    = -42
            b5    = 10
            b6    = 12
            c1    = 14
            c2    = -37
            c3    = 19
            c4    = 27

            numerator = b0*(1+b1*x+b2*x**2.+b3*x**3.+b4*x**4.+b5*x**5.+b6*x**6.)
            denominator = 1+c1*x+c2*x**2.+c3*x**3.+c4*x**4.

            PSD = S0*(x**alpha + a0*x**beta + numerator/denominator)


        elif specs['survey'] == 'LISA':
           print('Coming soon...')
           exit()
        else:
           print('ERROR: Unknown survey')
           exit()

        return PSD

    def survey_specs(self,specs,GWpars,total_mass):
    
        theta,phi = self.GWdist.sky_distribution(GWpars)

        pol = 0. #MM: this is the polarization angle. Needs to be reviewed

        if specs['survey'] == 'Einstein Telescope':
       
           F_plus  = []
           F_cross = []

           #Working on the 3 antenna patterns from ET
           coeff = 0
           for ind in range(3):
               plus, cross = self.antenna_pattern(theta,phi+coeff*np.pi/3.,pol)
               F_plus.append(plus)
               F_cross.append(cross)
               coeff += 2

       
           f_lower = 1                              # Hz; lower limit for inner product integral
           f_upper = 2/((6**(3/2))*2*np.pi*total_mass) # upper limit for inner product integral #MM: units of this to be checked!!!!


        elif options['survey'] == 'LISA':
           print('Coming soon...')
           exit()
        else:
           print('ERROR: Unknown survey')
           exit()


        return F_plus,F_cross,f_lower,f_upper
