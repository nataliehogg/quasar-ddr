import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

from .SMBH_specifications import SMBHSpecs

clight = 299792.458


class RingLuminosity:

    def __init__(self, cosmoparams, specifications, z_in, outpath, luminosity_distance):

        print('Generating SMBH luminosity distance mock.')

        # initialise class
        SMBH = SMBHSpecs()

        # get the redshift distribution
        self.redshifts = SMBH.redshift_distribution(specifications)

        # NH: is there a better way to deal with the units here?

        d_L   = interp1d(z_in, clight/cosmoparams['H0']*luminosity_distance(z_in))

        err_1 = 0.01*(clight/cosmoparams['H0'])
        err_2 = 0.02*(clight/cosmoparams['H0'])
        err_3 = 0.12*(clight/cosmoparams['H0'])

        self.error   = np.sqrt(err_1**2+err_2**2+err_3**2) # NH: this needs to be made more realistic

        #Since we have only few points, checking here that the mock is "good"
        #Here we have lots of points, this should be useless
        muGW  = 1
        sdGW  = 3
        count = 0

        while (np.abs(muGW)>0.01) or (np.abs(sdGW-1)>0.01):
            #   hist = np.array([(self.data[ind]-d_L(zi))/self.error[ind] for ind,zi in enumerate(self.redshifts)])
              self.data = [d_L(zi)+np.random.normal(loc=0.0, scale=self.error) for ind, zi in enumerate(self.redshifts)]
              hist = np.array([(self.data[ind]-d_L(zi))/self.error for ind,zi in enumerate(self.redshifts)])
              muGW, sdGW = norm.fit(hist)
              count += 1

        # print('Saving mock to '+outpath+'_dl_dataset.txt')

        # np.savetxt(outpath+'_dl_dataset.txt', np.array([self.redshifts, 
        #                                                 self.data,
        #                                                 self.error]).T, 
        #                                                 delimiter=' ')

    def mock(self):

        mockdict = {}
        mockdict['z']     = self.redshifts
        mockdict['value'] = self.data
        mockdict['error'] = self.error

        return mockdict