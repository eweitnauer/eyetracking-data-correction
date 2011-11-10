import datasource as DS
import scipy as S


class EyeTrackerFakeDataSource(DS.SeededDataSource):
    '''A data source that fakes fixation data for a static scene.
    
    At certain locations
    '''
    def __init__(self, locs=[], sigmas=[], base_probabilities=[], covariances=[],
                 dt=200, sigma_dt=50, **kws):
        '''
        @param locs:
            A list with locations [ (x0,y0), (x1,y1), ..., (xn,xy) ].
        @param sigmas:
            A list with standard deviation values for the (symmetric) Gaussian blobs.
        @base_probabilities:
            A list with the probabilities for each blob.
            The list will be normalized on __init__.
            Defaults: [ 1/n, 1/n, ..., 1/n ]
        @param covariances: 
            Optional (if the Gaussians blobs should not be symmetric), you
            can assign a list of covariance matrices.
            See scipy.random.multivariate_normal.
        @param dt: 
            The delta t in milliseconds from one fixation to the next.
        @param sigma_dt:
            In order to jitter the t-positions a bit, we draw the delta-t from
            a normal distribution with mean=dt and sigma=sigma_dt.
        '''
        n = len(locs)
        self.locs = locs
        if sigmas == [] or sigmas is None:
            self.sigmas = [1] * n
        else:
            assert len(sigmas) == n
            self.sigmas = sigmas
        
        if base_probabilities == [] or base_probabilities is None:
            self._cs = S.arange(n)+1
        else:
            assert len(base_probabilities) == n
            self._cs = S.cumsum(base_probabilities)
        
        if covariances == [] or covariances is None:
            # Defaults to circular blog: identity matrix
            self.covariances = [S.array( [[1,0],[0,1]] ) ] * n
        else:
            assert len(covariances) == n
            self.covariances = S.array(covariances)

        self.dt = dt
        self.sigma_dt = sigma_dt
        
        super(EyeTrackerFakeDataSource, self).__init__(output_dim=3, **kws)
        self._reset()
        
        
    def _reset(self):
        self._t = 0
    
    
    def _sample(self):
        # choose the right Gaussian by using the base_probabilities:
        r  = self.random.uniform(low=0.0, high=self._cs[-1])
        for i in range(len(self.locs)):
            if self._cs[i] > r: break
        # And now we draw from the i-th Gaussian
        x,y = self.random.multivariate_normal( mean=self.locs[i],
                                               cov=self.sigmas[i]**2 * self.covariances[i])
        dt = self.random.normal(loc=self.dt, scale=self.sigma_dt)
        self._t += dt
        return [self._t, x, y] # return [ [T,X,Y] ]

