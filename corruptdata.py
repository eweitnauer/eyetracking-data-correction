'''
Bratty distortion of 2D eyetracking data. 

@copyright: 
    2011 Samuel John
@author: 
    Samuel John
@contact: 
    www.SamuelJohn.de
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

import mdp
import scipy as S
from eyetracker_data import T, X, Y

class ShiftEyeTrackingData(mdp.Node):
    '''
    Add linear shift in x/y direction to a an EyeTrackerDataSource's output.
    '''
    def __init__(self, shift_x=0, shift_y=0, t0=0, t1=1, **kws):
        '''A linear shift is applied to the data: at t <= t0 the shift
        is 0, at t >= t1 the shift (shift_x, shift_y). Between t0 and t1 it is
        linearely interpolated.'''
        super(ShiftEyeTrackingData, self).__init__(**kws)
        self.shift_x = shift_x
        #self.slope_x = shift_x / (t1-t0)
        self.shift_y = shift_y
        #self.slope_y = shift_y / (t1-t0)
        self.t0 = t0
        self.t1 = t1
        assert t0 <= t1

    def is_trainable(self): return False

        
    def _execute(self, data):
        slope_x = self.shift_x / (self.t1-self.t0)
        slope_y = self.shift_y / (self.t1-self.t0)
        ts = data[:,T]
        where = S.where(ts > self.t0)
        dt_max = self.t1 - self.t0
        if self.shift_x != 0:
            data[where,X] += S.minimum(data[where,T]-self.t0, dt_max) * slope_x
        if self.shift_y != 0:
            data[where,Y] += S.minimum(data[where,T]-self.t0, dt_max) * slope_y
        return data
    
    def __str__(self):
        return self.__class__.__name__ + '(shift_x=%.2f, shift_y=%.2f, t0=%.0f, t1=%.0f'%(self.shift_x, self.shift_y, self.t0, self.t1)
    __repr__ = __str__


class JerkEyeTrackingData(mdp.Node):
    '''
    Add a jumpt in x/y direction at time t to an EyeTrackerDataSource's output.
    '''
    def __init__(self, jerk_at_t=0, jerk_x=0, jerk_y=0, **kws):
        super(JerkEyeTrackingData, self).__init__(**kws)
        self.jerk_at = jerk_at_t
        self.jerk_x  = jerk_x
        self.jerk_y  = jerk_y
    
    def is_trainable(self): return False    
        
    def _execute(self, data):
        where = S.where(data[:,T] >= self.jerk_at)
        if self.jerk_x != 0:
            data[where,X] += self.jerk_x
        if self.jerk_y != 0:
            data[where,Y] += self.jerk_y
        return data

    def __str__(self):
        return self.__class__.__name__ + '(jerk_at_t=%.0f, jerk_x=%.2f, jerk_y=%.2f'%(self.jerk_at_t, self.jerk_x, self.jerk_y)
    __repr__ = __str__

