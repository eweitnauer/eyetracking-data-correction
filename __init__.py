from eyedata import FixationData, FixationDataFromCSV, EyeTrackerDataSource
from fakedata import EyeTrackerFakeDataSource
from corruptdata import ShiftEyeTrackingData, JerkEyeTrackingData

# constants for the columns in the samples of a EyeTrackerDataSource
T, X, Y = 0, 1, 2