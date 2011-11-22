'''
A datasource (based on https://github.com/samueljohn/datasource) that
reads CSV values from an eyetracking experiement and supplies the data in a
convenient manner.

There is a datasource that can "fake" eyetracking data for a quite simple case.
The ShiftEyeTrackingData and JerkEyeTrackingData can be used to simulate 
systematic errors in the datasource.

@author: 
    Samuel John, Erik Weitnauer
'''
from eyedata import T, X, Y, FixationData, FixationDataFromCSV, EyeTrackerDataSource
from fakedata import EyeTrackerFakeDataSource
from corruptdata import ShiftEyeTrackingData, JerkEyeTrackingData
