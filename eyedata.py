
import datasource as DS
import scipy as S
import logging
import os

class FixationData(object):
    ''''Base class for fixation data for several trials over several persons.'''
    def __init__(self):
        self.trials  = {}
        
    def query(self, trial_id, person_id, eye='L'):
        return self.trials[trial_id][person_id][eye]
        
        

class FixationDataFromCSV(FixationData):
    '''Load CVS file and prepare the data for pythonic access.'''
    def __init__(self, filename="fixation_data.csv", delimiter=",", skiprows=1, **kws):
        if not os.path.isabs(filename):
            filename = os.path.dirname(os.path.abspath(__file__))+os.sep+filename
        super(FixationDataFromCSV, self).__init__(**kws)
        log = logging.getLogger('FixationData')
        log.info('Loading %s ...', filename)
        # manually unpack for numpy 1.5 compatibility
        temp = S.loadtxt(filename, 
                         delimiter=delimiter, 
                         skiprows=skiprows,
                         unpack=False,
                         dtype=[ ('trail',S.int32),
                                 ('person',S.int32),
                                 ('eye','S1'), #string of len 1
                                 ('t',S.float32),
                                 ('x',S.float32),
                                 ('y',S.float32)])
        trial, person, eye, t, x, y = (temp[n] for n in temp.dtype.names)
        log.info('Found %i entries', trial.size)
        for trial_id in S.unique(trial):
            self.trials[trial_id] = {}
            for person_id in S.unique(person):
                self.trials[trial_id][person_id] = {}
                for eye_id in ('L','R'):
                    idx = (trial == trial_id) & (person == person_id) & (eye == eye_id)
                    self.trials[trial_id][person_id][eye_id] = S.vstack((t[idx], x[idx], y[idx])).T  
        log.info('Loaded trials: ' + str(self.trials.keys()))



class EyeTrackerDataSource(DS.DataSource):
    '''A data source for the trial of a single person for one eye.'''
    def __init__(self, fixation_data=None, trial_id=0, person_id=0, eye='L', **kws):
        self.data      = fixation_data
        self.trial_id  = trial_id
        self.person_id = person_id
        self._reset()
        self.eye       = eye
        if eye is None:
            dL = self.data.query(trial_id=trial_id,person_id=person_id,eye='L')
            dR = self.data.query(trial_id=trial_id,person_id=person_id,eye='R')
            if len(dL) > len(dR):
                self.eye = 'L'
            else:
                self.eye = 'R'
        d = self.data.query(trial_id=trial_id,person_id=person_id,eye=self.eye)
        assert d.ndim == 2
        super(EyeTrackerDataSource, self).__init__(number_of_samples_max=len(d), 
                                                   output_dim=d.shape[1],
                                                   **kws)
        
        
    def _reset(self):
        self._t = 0
        
        
    def _samples(self, n=1):
        s = self.data.query(trial_id=self.trial_id, 
                            person_id=self.person_id, 
                            eye=self.eye)[self._t:self._t+n]
        self._t += n
        return s.copy()
    
    
    def __repr__(self):
        if self.name is None:
            name = ''
        else:
            name = str(self.name)
        return '<DataSource %(name)s oudput_dim=%(output_dim)i with %(n)i/%(max)s samples, trial=%(trial)i, person=%(person)i, eye=%(eye)s>' \
               % dict(name=name, 
                      output_dim=self.output_dim, 
                      n=self.number_of_samples_until_now, 
                      max=str(self.number_of_samples_max), 
                      trial=self.trial_id, 
                      person=self.person_id,
                      eye=str(self.eye))
    __str__ = __repr__
    
    

class EyeTrackerDataSourceWithShift(EyeTrackerDataSource):
    def __init__(self, shift_function, **kws):
        super(EyeTrackerDataSourceWithShift,self).__init__(**kws)
        self.shift_function = shift_function
        
    def _samples(self, n=1):
        s = EyeTrackerDataSource._samples(self, n=n)
        s = self.shift_function(s)
        return s
        
        
        

    
    
    
    
if __name__ == '__main__':
    d = FixationDataFromCSV()
    #print(d.trials)
    DS = EyeTrackerDataSource(fixation_data=d, trial_id=7, person_id=5, eye=None)
    print(repr(DS))



