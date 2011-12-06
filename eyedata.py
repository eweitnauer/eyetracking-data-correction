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
        '''Return all entries for the given trial_id (int) person_id (int) and 
        eye (str of len 1).'''
        return self.trials[trial_id][person_id][eye]
    
    def get_imagepath(self, trial_id, person_id):
        return self.trials[trial_id][person_id]['img_filename']
        
        

class FixationDataFromCSV(FixationData):
    '''Load CSV file and prepare the data for pythonic access.
    
    The format of the CSV file should be (iff image_dir is set):
       trail, person, img_filename, eye,               time,  x,    y 
       int,   int,    string,      'L' or 'R' or 'X',  float, float, float
    or:
       trail, person, eye,               time,  x,    y 
       int,   int,    'L' or 'R' or 'X',  float, float, float
    
    @note:
        The eye 'X' does not denote a fixation but the ground truth information
        of the true object location to look at for that trial/person. At least
        this is our interpretation for one of our data sets.
    '''
    def __init__(self, filename="fixation_data_pbp.csv", delimiter=",", skiprows=1,
                 ranges=[[-100,1200],[600, -100]], image_dir=None, **kws):
        if not os.path.isabs(filename):
            filename = os.path.dirname(os.path.abspath(__file__))+os.sep+filename
        super(FixationDataFromCSV, self).__init__(**kws)
        log = logging.getLogger('FixationData')
        log.info('Loading %s ...', filename)
        
        if image_dir is not None:
            self.images_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep + image_dir + os.sep
            
            # manually unpack for numpy 1.5 compatibility
            temp = S.loadtxt(filename, 
                             delimiter=delimiter, 
                             skiprows=skiprows,
                             unpack=False,
                             dtype=[ ('trail',S.int32),
                                     ('person',S.int32),
                                     ('img_filename','S40'),
                                     ('eye','S1'), #string of len 1
                                     ('t',S.float32),
                                     ('x',S.float32),
                                     ('y',S.float32)])
            trial, person, img_filename, eye, t, x, y = (temp[n] for n in temp.dtype.names)
            log.info('Found %i entires', trial.size)
            #self.trials['all'] = {} # special dict for all trials
        else:
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
            img_filename = None
            log.info('Found %i entires', trial.size)
            
        for trial_id in S.unique(trial):
            self.trials[trial_id] = {}
            for person_id in S.unique(person):
                self.trials[trial_id][person_id] = {}
                idx = (trial == trial_id) & (person == person_id)
                #idx_all_trials = (person == person_id)
                for eye_id in ('L','R','X'):
                    idx2 = idx & (eye == eye_id)
                    #idx2_all_trials = idx_all_trials & (eye == eye_id)
                    self.trials[trial_id][person_id][eye_id] = S.vstack((t[idx2], x[idx2], y[idx2])).T
                    # special handling for trail_id == 'all'
                    #self.trials['all'][person_id][eye_id] = S.vstack((t[idx2_all_trials], x[idx2_all_trials], y[idx2_all_trials])).T
                if img_filename is not None and len(img_filename[idx]) > 0: 
                    self.trials[trial_id][person_id]['img_filename'] = img_filename[idx][0]
                
        if ranges is None:
            self.ranges = [[x.min(), x.max()]
                          ,[y.min(), y.max()]]
        else:
            assert len(ranges) == 2 and len(ranges[0]) == 2
            self.ranges = ranges
                    
        log.info('Loaded trials: ' + str(self.trials.keys()))
        


class EyeTrackerDataSource(DS.DataSource):
    '''A data source for the trial of a single person for one eye.'''
    def __init__(self, fixation_data=None, trial_id=0, person_id=0, eye=None, ranges=None, **kws):
        self.data   = fixation_data
        self.ranges = ranges if ranges is not None else fixation_data.ranges
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
        return '<DataSource %(name)s oudput_dim=%(output_dim)i with /%(max)s fixations, trial=%(trial)i, person=%(person)i, eye=%(eye)s>' \
               % dict(name=name, 
                      output_dim=self.output_dim, 
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
        
    

# Unfortunately we have to define a new DS to handle the ground truth data that 
# is encoded as the eye "X"
class FixationDataFromCSVwithGroundTruth(FixationDataFromCSV):    
    def __init__(self, ranges=[[-100,2000],[2000, -100]], **kws):
        super(FixationDataFromCSVwithGroundTruth, self).__init__(image_dir=None, **kws) # will do the parsing    
        all_trial_ids = sorted(self.trials.keys())
        new_trials = {}
        new_trials[0] = {} # there will be only trial 0 (no other trials!)
        all_person_ids = sorted(self.trials[0].keys()) # in a trial are the persons stored
        for pid in all_person_ids:
            new_trials[0][pid] = {}
            new_trials[0][pid]['L'] = []
            new_trials[0][pid]['R'] = []
            new_trials[0][pid]['X'] = []
            
        for tid in all_trial_ids:
            for pid in sorted(self.trials[tid].keys()):
                # get the true locations 
                gt = S.array(self.query(tid, person_id=pid, eye='X'))
                for eye_id in ('L','R'):
                    tmp = S.array(self.query(tid, person_id=pid, eye=eye_id))
                    if len(tmp) > 0:
                        t_max = S.argmax(tmp[:,0])
                        
                        # take only the last fixation in this trial=tid
                        # and set the time to the time of the gt
                        lt = new_trials[0][pid][eye_id][-1][0] if len(new_trials[0][pid][eye_id]) >0 else 0
                        #gt[0][0] += lt
                        new_trials[0][pid][eye_id].append( [gt[0][0]+lt, tmp[t_max][1], tmp[t_max][2] ] )  
                        #                                  time of gt, the last fixations location x,y
                        new_trials[0][pid]['X'].append( [gt[0][0]+lt, gt[0][1], gt[0][2]] )
        for pid in new_trials[0].keys():
            for eye_id in new_trials[0][pid].keys():
                new_trials[0][pid][eye_id] = S.array(new_trials[0][pid][eye_id]) 
        self._old_trials = self.trials
        self.trials = new_trials 
        # for each trial (of the 200) find the last fixation
        # and the ground truth "X"
        # ground_truth defines wich kind is returned by self.query
        # self.ranges should be the same for both    
    
if __name__ == '__main__':
    d = FixationDataFromCSV(filename="fixation_data_pbp.csv", image_dir='pbp_imgs')
    #print(d.trials)
    DS = EyeTrackerDataSource(fixation_data=d, trial_id=7, person_id=5, eye=None)
    for i in range(6):
        print DS.sample()
    print(repr(DS))



