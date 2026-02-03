import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class MockChecker:
    """
    Validate and visualize mock data quality.

    This class performs quality checks on generated mock data by comparing
    to fiducial model predictions, computing outliers, and creating diagnostic
    plots to verify the data follows expected statistical distributions.

    Parameters
    ----------
    x : array_like
        Independent variable (e.g., redshift)
    data : array_like
        Mock data values
    error : array_like or float
        Observational errors (can be array or scalar)
    fiducial : array_like
        Fiducial model predictions
    threshold : float
        Outlier threshold in units of sigma
    visualize : bool, optional
        Whether to create visualizations (default: False)
    dataname : str, optional
        Name for plots and reports (default: 'test')

    Attributes
    ----------
    outliers : dict
        Dictionary containing outlier information
    """

    def __init__(self, x=None, data=None, error=None, fiducial=None,
                 threshold=None, visualize=False, dataname='test'):

        self.x         = x
        self.data      = data
        self.error     = error
        self.fiducial  = fiducial
        self.threshold = threshold



        print('-----------------------------------REPORT ON '+dataname+'-----------------------------------')

        self.outliers = self.get_outliers()

        print('Number of outliers: '+str(len(self.outliers['x'])))
        print('Fraction of data: '+str(100*len(self.outliers['x'])/len(self.x))+'%')
        if len(self.outliers['x'])>0:
            print('Maximum distance from fiducial (in sigma): '+str(np.sort(self.outliers['distance'])[-1]))


        self.histogram(dataname)
        self.plot_mock(dataname)


    def histogram(self,name):

        # NH: this issue of indexing the error array must be understood 

        # hist = np.array([(self.data[ind]-self.fiducial[ind])/self.error[ind] for ind in range(len(self.x))])
        hist = np.array([(self.data[ind]-self.fiducial[ind])/self.error for ind in range(len(self.x))])

        mu,sigma = norm.fit(hist)
        x = np.linspace(-10,10,100)

        limit = max(np.abs(min(hist)),max(hist))

        plt.figure(figsize=(10, 8))
        plt.title(name+' histogram', fontsize=20)
        plt.hist(hist, density=True, bins='auto', edgecolor='black', linewidth=2)  # `density=False` would make counts
        plt.plot(x,norm.pdf(x,mu,sigma),ls='-',linewidth=3,color='#FFB300',label=r'$\mu=$'+str(mu)+r' $\sigma=$'+str(sigma))
        plt.ylabel('Counts', fontsize=20)
        plt.xlabel(r'(Data-fiducial)/$\sigma$', fontsize=20)
        plt.axvline(0,ls='--',lw=3,color='#8E001C')
        plt.xlim([-3*limit,3*limit])
        plt.legend(loc='upper right')
        plt.show()

        return None

    def plot_mock(self,name):

        plt.figure(figsize=(10, 8))
        plt.title(name+' mock',fontsize=20)
        plt.plot(self.x,self.fiducial,label=r'Fiducial',ls='-',color='#8E001C')
        plt.errorbar(self.x,self.data,yerr=self.error,label='data',ls='',marker='*',color='#FFB300')
        plt.errorbar(self.outliers['x'],self.outliers['data'],
                     label=r' '+str(self.threshold)+r'$\sigma$ outliers ('+str(100*len(self.outliers['x'])/len(self.x))+'% of total)',
                     ls='',marker='x',markersize=10,color='black')
        plt.xlabel(r'$x$',fontsize=20)
        plt.ylabel(r'$y(x)$',fontsize=20)
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left",fontsize=20)
        plt.show()


        
    def get_outliers(self):

        outliers = {}


        distance       = np.abs(self.data-self.fiducial)/self.error

        def condition(x): return x > self.threshold

        outliers['x']        = np.array([self.x[idx] for idx, element in enumerate(distance) if condition(element)])
        outliers['data']     = np.array([self.data[idx] for idx, element in enumerate(distance) if condition(element)])
        # outliers['error']    = np.array([self.error[idx] for idx, element in enumerate(distance) if condition(element)])
        outliers['error']    = np.array([self.error for idx, element in enumerate(distance) if condition(element)])
        outliers['distance'] = np.array([element for  element in distance if condition(element)])

        return outliers




    





