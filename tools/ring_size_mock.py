import numpy as np
from scipy.stats import norm

from .SMBH_specifications import SMBHSpecs

clight = 299792.458


class RingSize:

    def __init__(self, cosmoparams, specifications, z_in, outpath, angular_distance):

        print('Generating SMBH angular distance mock.')

        # initialise class
        SMBH = SMBHSpecs()

        # get the redshift distribution
        self.redshifts = SMBH.redshift_distribution(specifications)

        # NH: is there a better way to deal with the units here?

        angular   = angular_distance

        err_1 = 0.01*(clight/cosmoparams['H0'])
        err_2 = 0.02*(clight/cosmoparams['H0'])
        err_3 = 0.12*(clight/cosmoparams['H0'])

        self.error   = np.sqrt(err_1**2+err_2**2+err_3**2) # NH: this needs to be made more realistic

        # dlarray = np.array([d_L(z) for z in self.redshifts])

        #Since we have only few points, checking here that the mock is "good"
        #Here we have lots of points, this should be useless
        muGW  = 1
        sdGW  = 3
        count = 0

        while (np.abs(muGW)>0.01) or (np.abs(sdGW-1)>0.01):
            #   self.da_data = [angular(zi)+np.random.normal(loc=0.0, scale=self.error[ind]) for ind, zi in enumerate(self.redshifts)]
            #   hist = np.array([(self.da_data[ind]-angular(zi))/self.error[ind] for ind, zi in enumerate(self.zBAO)])
              self.data = [angular(zi)+np.random.normal(loc=0.0, scale=self.error) for ind, zi in enumerate(self.redshifts)]
              hist = np.array([(self.data[ind]-angular(zi))/self.error for ind, zi in enumerate(self.redshifts)])
              muGW,sdGW = norm.fit(hist)
              count += 1

        # while (np.abs(muGW)>0.01) or (np.abs(sdGW-1)>0.01):
        #       #Gaussian spread of data points using error
        #       #Gaussian is truncated to avoid dL<0 points
        #       self.data = np.zeros((len(self.redshifts)))
        #       for ind in range(len(self.data)):
        #           mean   = dlarray[ind]
        #           stdev  = self.error#[ind]
        #           lowlim = 0.
        #           upplim = 1.e6
        #           self.data[ind] = truncnorm.rvs((lowlim-mean)/stdev, (upplim-mean)/stdev, loc=mean, scale=stdev, size=1)

        #     #   hist = np.array([(self.data[ind]-d_L(zi))/self.error[ind] for ind,zi in enumerate(self.redshifts)])
        #       hist = np.array([(self.data[ind]-d_L(zi))/self.error for ind,zi in enumerate(self.redshifts)])
        #       muGW, sdGW = norm.fit(hist)
        #       count += 1

        # print('Saving mock to '+outpath+'_da_dataset.txt')

        # np.savetxt(outpath+'_da_dataset.txt', np.array([self.redshifts, 
        #                                                 self.data,
        #                                                 self.error]).T, 
        #                                                 delimiter=' ')

    def mock(self):

        mockdict = {}
        mockdict['z']     = self.redshifts
        mockdict['value'] = self.data
        mockdict['error'] = self.error

        return mockdict