'''
Fake data to simulate and evaluate eye tracking correction.

@copyright: 
    2011 Samuel John and Erik Weitnauer
@author: 
    Samuel John and Erik Weitnauer
@contact: 
    www.SamuelJohn.de, eweitnauer@gmail.com
@license: 
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''
__version__ = 1.0

import datasource as DS
import scipy as S


class EyeTrackerFakeDataSource(DS.SeededDataSource):
    '''A data source that fakes fixation data for a static scene.
    
    At certain locations (locs), gaussian blobs are used to generate new
    fixation data. First a gaussian blob is chosen (with base_probabilities)
    and then the sigmas are used to get scale the blob. 
    Optionally the Gaussians can have a covariance that makes them elongated
    or rotated.
    '''
    def __init__(self, locs=[], sigmas=[], base_probabilities=[], covariances=[],
                 dt=200, sigma_dt=50, ranges=None, uniform_random_fixations_probability=0.1, **kws):
        '''
        @param locs:
            A list with locations [ (x0,y0), (x1,y1), ..., (xn,xy) ].
        @param sigmas:
            A list with standard deviation values for the (symmetric) Gaussian blobs.
            Defaults to [1,1,1,...,1]
        @base_probabilities:
            A list with the probabilities for each blob.
            The list will be normalized on __init__.
            Defaults: [ 1/n, 1/n, ..., 1/n ]
        @param covariances: 
            Optional. If the Gaussians blobs should not be symmetric, you
            can assign a list of covariance matrices.
            See scipy.random.multivariate_normal.
        @param dt: 
            The delta t in milliseconds from one fixation to the next. 
            Default to 200.
        @param sigma_dt:
            In order to jitter the t-positions a bit, we draw the delta-t from
            a normal distribution with mean=dt and sigma=sigma_dt.
            Default to 50.
        @param ranges:
            [[xmin, xmax],[ymin,ymax]], if not set, it is automatically set to
            the actual data range of the locations plus 4*sigma.
        '''
        n = len(locs)
        self.locs = S.array(locs)
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
        
        if ranges is None:
            margin = max(self.sigmas) * 4
            self.ranges = [[self.locs[:,0].min()-margin, self.locs[:,0].max()+margin]
                          ,[self.locs[:,1].min()-margin, self.locs[:,1].max()+margin]]
        else:
            assert len(ranges) == 2 and len(ranges[0]) == 2
            self.ranges = ranges

        self.dt = dt
        self.sigma_dt = sigma_dt
        
        self.uniform_random_fixations_probability = uniform_random_fixations_probability
        
        super(EyeTrackerFakeDataSource, self).__init__(output_dim=3, **kws)
        self._reset()
        
        
    def _reset(self, **kws):
        super(EyeTrackerFakeDataSource,self)._reset(**kws)
        self._t = 0
    
    
    def _sample(self):
        xr = self.ranges[0]
        yr = self.ranges[1]
        dt = S.absolute(self.random.normal(loc=self.dt, scale=self.sigma_dt))
        self._t += dt
        # First check if we create a new uniform random fixation
        if self.random.uniform() < self.uniform_random_fixations_probability:
            x = self.random.uniform(low=xr[0], high=xr[1])
            y = self.random.uniform(low=yr[0], high=yr[1])
            return [self._t, x, y] # return [ [T,X,Y] ]
        # choose the right Gaussian by using the base_probabilities:
        r  = self.random.uniform(low=0.0, high=self._cs[-1])
        for i in range(len(self.locs)):
            if self._cs[i] > r: break
        # And now we draw from the i-th Gaussian
        while True:
            x,y = self.random.multivariate_normal( mean=self.locs[i],
                                                   cov=self.sigmas[i]**2 * self.covariances[i])
            if not(xr[0] < x < xr[1]): continue
            if not(yr[0] < y < yr[1]): continue
            break
        return [self._t, x, y] # return [ [T,X,Y] ]

    def __str__(self):
        locs = ', '.join( [ "("+format(x,".0f") +','+ format(y,".0f")+")" for x,y in self.locs] )
        sigmas = ','.join( [ format(s,".1f") for s in self.sigmas] )
        return self.__class__.__name__ + "(locs=[%s], sigmas=[%s], dt=%.1f)" %(locs, sigmas, self.dt )
        
    __repr__ = __str__    
