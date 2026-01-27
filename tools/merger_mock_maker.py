'''
creating mock merger data 22/01/2020
'''

# import general stuff
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random as r
from scipy.integrate import quad
from scipy.stats import truncnorm
from scipy.stats import norm
from tools.mergers_specifications import *
from tools.noise_calculation import *


class compute_GWmock:

    def __init__(self,cosmoparams,GWparams,specifications,outpath,Hubble,comoving_distance,luminosity_distance):


        print('Welcome to the GW standard sirens mock maker!')
        print('')
        print('output files will be produced in '+outpath+'_GWdataset.txt')

        d_C     = comoving_distance
        d_L     = luminosity_distance
        H       = Hubble

        #Initializing classes
        self.GWdist = GWdist(d_C,H)
        self.survey = survey(self.GWdist)
        self.noise  = noise(self.survey)

        # Getting redshift distribution
        self.redshifts = self.GWdist.redshift_distribution(specifications,GWparams)

        # Getting masses
        total_mass, chirp_mass, eta = self.GWdist.masses(GWparams['binary_type'],self.redshifts)

        # Computing error
        self.error = self.noise.compute_noise(cosmoparams,specifications,GWparams,self.redshifts,d_L,total_mass,chirp_mass,eta)


        dlarray = np.array([d_L(z) for z in self.redshifts])

        #Since we have only few points, checking here that the mock is "good"
        #Here we have lots of points, this should be useless
        muGW  = 1
        sdGW  = 3
        count = 0

        while (np.abs(muGW)>0.01) or (np.abs(sdGW-1)>0.01):
              #Gaussian spread of data points using error
              #Gaussian is truncated to avoid dL<0 points
              self.data = np.zeros((len(self.redshifts)))
              for ind in range(len(self.data)):
                  mean   = dlarray[ind]
                  stdev  = self.error[ind]
                  lowlim = 0.
                  upplim = 1.e6
                  self.data[ind] = truncnorm.rvs((lowlim-mean)/stdev, (upplim-mean)/stdev, loc=mean, scale=stdev, size=1)

              hist = np.array([(self.data[ind]-d_L(zi))/self.error[ind] for ind,zi in enumerate(self.redshifts)])
              muGW,sdGW = norm.fit(hist)
              count += 1


        with open(os.path.join(os.getcwd(), outpath+'_GWdataset.txt'), 'w') as f:
             f.write('#z dL sigma_dL\n')

             for ind in range(len(self.redshifts)):
                 f.write("%f " %self.redshifts[ind] + " %f" %self.data[ind] +  " %f" %self.error[ind] + " \n")

        f.close()

        print('')
        print('Mock done and saved')


    def mock(self):

        mockdict = {}
        mockdict['z']     = self.redshifts
        mockdict['value'] = self.data
        mockdict['error'] = self.error
        
        return mockdict


    def zdist(self,specs,GWpars):

        dist = self.GWdist.probability(specs,GWpars)

        return dist

   
