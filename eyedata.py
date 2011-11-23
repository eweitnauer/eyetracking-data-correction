'''
Load CSV data and provide a datasource for the samples in there.
The returned values from the datasource are [t,x,y], each.

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
import mdp
import logging
import os

# constants for the columns in the samples of a EyeTrackerDataSource
T, X, Y = 0, 1, 2


class FixationData(object):
    ''''Base class for fixation data for several trials over several persons.'''
    def __init__(self):
        self.trials  = {}
        
        
    def query(self, trial_id, person_id, eye='L'):
        return self.trials[trial_id][person_id][eye]
    
    def get_imagepath(self, trial_id, person_id):
        return self.trials[trial_id][person_id]['img_filename']
        
        

class FixationDataFromCSV(FixationData):
    '''Load CSV file and prepare the data for pythonic access.
    
    The format of the CSV file should be:
       trail, person, file_name, eye,        time,  x,     y 
       int,   int,    string,   'L' or 'R',  float, float, float
    '''
    def __init__(self, filename="fixation_data.csv", delimiter=",", skiprows=1, **kws):
        if not os.path.isabs(filename):
            filename = os.path.dirname(os.path.abspath(__file__))+os.sep+filename
        super(FixationDataFromCSV, self).__init__(**kws)
        log = logging.getLogger('FixationData')
        log.info('Loading %s ...', filename)
        
        self.images_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'imgs' + os.sep
        
        # manually unpack for numpy 1.5 compatibility
        temp = S.loadtxt(filename, 
                         delimiter=delimiter, 
                         skiprows=skiprows,
                         unpack=False,
                         dtype=[ ('trail',S.int32),
                                 ('person',S.int32),
                                 ('filename','S40'),
                                 ('eye','S1'), #string of len 1
                                 ('t',S.float32),
                                 ('x',S.float32),
                                 ('y',S.float32)])
        trial, person, filename, eye, t, x, y = (temp[n] for n in temp.dtype.names)
        log.info('Found %i entires', trial.size)
        for trial_id in S.unique(trial):
            self.trials[trial_id] = {}
            for person_id in S.unique(person):
                self.trials[trial_id][person_id] = {}
                idx = (trial == trial_id) & (person == person_id)
                for eye_id in ('L','R'):
                    idx2 = idx & (eye == eye_id)
                    self.trials[trial_id][person_id][eye_id] = S.vstack((t[idx2], x[idx2], y[idx2])).T
                if len(filename[idx]) > 0: 
                    self.trials[trial_id][person_id]['img_filename'] = filename[idx][0]
        log.info('Loaded trials: ' + str(self.trials.keys()))
        


class EyeTrackerDataSource(DS.DataSource):
    '''A data source for the trial of a single person for one eye.'''
    def __init__(self, fixation_data=None, trial_id=0, person_id=0, eye=None, **kws):
        self.data      = fixation_data
        self.trial_id  = trial_id
        self.person_id = person_id
        self._reset()
        self.T         = T
        self.X         = X
        self.Y         = Y
        self.eye      = eye
        if eye is None:
            dL = self.data.query(trial_id=trial_id,person_id=person_id,eye='L')
            dR = self.data.query(trial_id=trial_id,person_id=person_id,eye='R')
            if len(dL) > len(dR):
                self.eye = 'L'
            else:
                self.eye = 'R'
        d = self.data.query(trial_id=trial_id,person_id=person_id,eye=self.eye)
        assert d.ndim == 2, d
        self.t_begin = d[0,self.T]
        self.t_end   = d[-1,self.T]
        super(EyeTrackerDataSource, self).__init__(number_of_samples_max=len(d), 
                                                   output_dim=d.shape[1],
                                                   **kws)
        
        
    def _reset(self):
        self._t = 0 # count the number of samples already drawn
        
        
    def _samples(self, n=1):
        s = self.data.query(trial_id=self.trial_id, 
                            person_id=self.person_id, 
                            eye=self.eye)[self._t:self._t+n]
        self._t += n 
        return s.copy()
    
    
    def get_imagepath(self):
        '''Get the complete absolute path to the image file for the this data source.'''
        return self.data.images_dir + self.data.get_imagepath(self.trial_id, self.person_id)
    
    
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

    def __str__(self):
        if self.name is None:
            name = ''
        else:
            name = str(self.name)
        return '<DataSource %(name)s trial=%(trial)i, pers=%(person)i %(eye)s>' \
               % dict(name=name, 
                      n=self.number_of_samples_until_now, 
                      trial=self.trial_id, 
                      person=self.person_id,
                      eye=str(self.eye))
        
    
    
    
if __name__ == '__main__':
    d = FixationDataFromCSV()
    #print(d.trials)
    DS = EyeTrackerDataSource(fixation_data=d, trial_id=7, person_id=5, eye=None)
    print(repr(DS))



