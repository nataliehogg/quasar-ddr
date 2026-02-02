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

        err_1 = 0.01*(clight/cosmoparams['H0'])

        # NH: as in the dL mock, this needs work
        self.error = np.ones(len(self.redshifts))*np.sqrt(err_1**2) # NH: this needs to be made more realistic

        #Since we have only few points, checking here that the mock is "good"
        #Here we have lots of points, this should be useless
        muGW  = 1
        sdGW  = 3
        count = 0

        while (np.abs(muGW)>0.01) or (np.abs(sdGW-1)>0.01):
              self.data = [angular_distance(zi)+np.random.normal(loc=0.0, scale=self.error[ind]) for ind, zi in enumerate(self.redshifts)]
              hist = np.array([(self.data[ind]-angular_distance(zi))/self.error[ind] for ind, zi in enumerate(self.redshifts)])
              muGW,sdGW = norm.fit(hist)
              count += 1

        print('Saving mock to '+outpath+'_da_dataset.txt')

        np.savetxt(outpath+'_da_dataset.txt', np.array([self.redshifts, 
                                                        self.data,
                                                        self.error]).T, 
                                                        delimiter=' ')

    def mock(self):

        mockdict = {}
        mockdict['z']     = self.redshifts
        mockdict['value'] = self.data
        mockdict['error'] = self.error

        return mockdict