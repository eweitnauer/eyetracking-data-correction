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
    def __init__(self, shift_x=0, shift_y=0, start_at_t=0, end_at_t=S.Infinity, **kws):
        '''The shifts are interpreted as x per time (and y per time).'''
        super(ShiftEyeTrackingData, self).__init__(**kws)
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.start_at_t = start_at_t
        self.end_at_t = end_at_t
        assert start_at_t <= end_at_t


    def is_trainable(self): return False

        
    def _execute(self, data):
        ts = data[:,T]
        where = S.where(self.start_at_t <= ts)
        dt_max = self.end_at_t - self.start_at_t
        if self.shift_x != 0:
            data[where,X] += S.minimum(data[where,T]-self.start_at_t, dt_max) * self.shift_x
        if self.shift_y != 0:
            data[where,Y] += S.minimum(data[where,T]-self.start_at_t, dt_max) * self.shift_y
        return data
    
    def __str__(self):
        return self.__class__.__name__ + '(shift_x: %f, shift_y: %f, start_at_t: %s, end_at_t: %s'%(self.shift_x, self.shift_y, self.start_at_t, self.end_at_t)
              


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

