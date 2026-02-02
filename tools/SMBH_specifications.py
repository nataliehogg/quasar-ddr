import numpy as np


class SMBHSpecs:

    def __init__(self):

        pass
    
    def redshift_distribution(self, specs):
        '''
        the redshifts for which we get mock luminosity distances and their errors
        '''

        redshifts = np.random.uniform(specs['z_min'], specs['z_max'], specs['n_SMBH']) # NH: this should be changed to something physically motivated

        return redshifts
