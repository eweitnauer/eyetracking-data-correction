import mdp
import scipy as S
from eyetracker_data import T, X, Y

class ShiftEyeTrackingData(mdp.Node):
    '''
    Add linear shift in x/y direction to a an EyeTrackerDataSource's output.
    '''
    def __init__(self, shift_x=0, shift_y=0, shift_start_at_t=0, shift_end_at_t=S.Infinity, **kws):
        '''The shifts are interpreted as x per time (and y per time).'''
        super(ShiftEyeTrackingData, self).__init__(**kws)
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_start_at_t = shift_start_at_t
        self.shift_end_at_t = shift_end_at_t
        self.T = T
        self.X = X
        self.Y = Y


    def is_trainable(self): return False

        
    def _execute(self, data):
        dT = data[:,T]
        where = S.where( (self.shift_start_at_t <= dT) * (dT <= self.shift_end_at_t))
        if self.shift_x > 0:
            data[where] += data[where] * self.shift_x
        if self.shift_y > 0:
            data[where] += data[where] * self.shift_y
        return data
              


class JerkEyeTrackingData(mdp.Node):
    '''
    Add a jumpt in x/y direction at time t to an EyeTrackerDataSource's output.
    '''
    def __init__(self, jerk_at_t=0, jerk_x=0, jerk_y=0, **kws):
        super(JerkEyeTrackingData, self).__init__(**kws)
        self.jerk_at = jerk_at_t
        self.jerk_x  = jerk_x
        self.jerk_y  = jerk_y
        self.T = T
        self.X = X
        self.Y = Y

    
    def is_trainable(self): return False    
        
    def _execute(self, data):
        # todo: do we need a copy here?
        where = S.where(data[:,T] >= self.jerk_at)
        if self.jerk_x > 0:
            data[where] += self.jerk_x
        if self.jerk_y > 0:
            data[where] += self.jerk_y
        return data

