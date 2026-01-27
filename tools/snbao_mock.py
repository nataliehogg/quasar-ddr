import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random as r
from scipy.integrate import quad
from scipy.stats import truncnorm
from scipy.stats import norm

clight = 299792.458

class compute_SNBAOmock:

    def __init__(self,cosmoparams,zin,outpath,Hubble,angular_distance,luminosity_distance,absolute=True):


        print('Welcome to the SN and BAO mock maker!')
        print('')
        print('output files will be produced in '+outpath+'_SNdataset.txt and '+outpath+'_BAOdataset.txt')



        #LSST MOCK SPECS
        Nsn = 8800
        zb = np.arange(0.1,1.1,0.1)
        mean_zb = [(zb[ind+1]+zb[ind])/2 for ind in range(len(zb)-1)]
        LSSTnum = [65,250,470,715,1100,1500,1700,2000,1000]

        LSSTz = []
        for ind in range(len(mean_zb)):
            mean   = mean_zb[ind]
            stdev  = 0.1
            lowlim = zb[ind]
            upplim = zb[ind+1]
            binz   =  truncnorm.rvs((lowlim-mean)/stdev, (upplim-mean)/stdev, loc=mean, scale=stdev, size=LSSTnum[ind])
            LSSTz.extend(binz)

        self.zsn   = np.sort(LSSTz)

        lumin   = clight/cosmoparams['H0']*luminosity_distance(zin) #

        if absolute:
           mufid = interp1d(zin,5*np.log10(lumin)+25,kind='cubic')
        else:
           M0 = -19.5 #CHECK ME!
           mufid = interp1d(zin,5*np.log10(lumin)+25+M0,kind='cubic')

        err_flux = 0.01
        err_scat = 0.02
        err_int  = 0.12
        delta_mu = np.array([np.random.normal(0,0.01)*z for z in self.zsn])

        self.errsn   = np.sqrt(err_flux**2+err_scat**2+err_int**2+delta_mu**2)


        #Since we have only few points, checking here that the mock is "good"
        #Here we have lots of points, this should be useless
        muSN  = 1
        sdSN  = 3
        count = 0


        while (np.abs(muSN)>0.01) or (np.abs(sdSN-1)>0.01):
              self.datasn = [mufid(zi)+np.random.normal(loc=0.0, scale=self.errsn[ind]) for ind,zi in enumerate(self.zsn)] 
              hist = np.array([(self.datasn[ind]-mufid(zi))/self.errsn[ind] for ind,zi in enumerate(self.zsn)])
              muSN,sdSN = norm.fit(hist)
              count += 1


        np.savetxt(outpath+'_SNdataset.txt', np.array([self.zsn,self.datasn,self.errsn]).T, delimiter=' ')



        #BAO mock (redshifts and errors taken from 1611.00036)

        self.zBAO   = np.array([0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85])
        rel_err_da = np.array([0.82,0.69,0.69,0.73,0.89,0.94,0.96,1.50,1.59,1.90,2.88,4.64,4.71])
        rel_err_H  = np.array([1.50,1.27,1.22,1.22,1.37,1.39,1.39,2.02,2.13,2.52,3.80,6.30,6.39])

        angular     = angular_distance
        hubble      = Hubble

        self.err_da = np.array([rel_err_da[ind]/100*angular(zi) for ind,zi in enumerate(self.zBAO)])
        self.err_H  = np.array([rel_err_H[ind]/100*hubble(zi) for ind,zi in enumerate(self.zBAO)])


        #Since we have only few points, checking here that the mock is "good"
        muda  = 1
        sdda  = 3
        count = 0

        while (np.abs(muda)>0.01) or (np.abs(sdda-1)>0.01):
              self.da_data = [angular(zi)+np.random.normal(loc=0.0, scale=self.err_da[ind]) for ind,zi in enumerate(self.zBAO)]
              hist = np.array([(self.da_data[ind]-angular(zi))/self.err_da[ind] for ind,zi in enumerate(self.zBAO)])
              muda,sdda = norm.fit(hist)
              count += 1

        print('Final DA mean = {}, stdev ={}'.format(muda,sdda))

        muH   = 1
        sdH   = 3
        count = 0

        while (np.abs(muH)>0.05) and (np.abs(sdH-1)>0.05):
              self.H_data  = [hubble(zi)+np.random.normal(loc=0.0, scale=self.err_H[ind]) for ind,zi in enumerate(self.zBAO)] 
              hist = np.array([(self.H_data[ind]-hubble(zi))/self.err_H[ind] for ind,zi in enumerate(self.zBAO)])
              muH,sdH = norm.fit(hist)
              count += 1
             
        print('Final H mean = {}, stdev ={}'.format(muH,sdH))
        

        np.savetxt(outpath+'_BAOdataset.txt', np.array([self.zBAO,self.da_data,self.err_da,self.H_data,self.err_H]).T, delimiter=' ')


    def SNmock(self):

        mockdict = {}
        mockdict['z']     = self.zsn
        mockdict['value'] = self.datasn
        mockdict['error'] = self.errsn

        return mockdict

    def BAOmock(self):

        mockdict = {}
        mockdict['z']        = self.zBAO
        mockdict['da_value'] = self.da_data
        mockdict['H_value']  = self.H_data
        mockdict['da_error'] = self.err_da
        mockdict['H_error']  = self.err_H

        return mockdict



