import os 

#os.path.dirname gets the directory of a path 

#the following variables can be imported: from tests import _PATH_DATA
_N_CLASSES = 10
_TEST_ROOT = os.path.dirname(__file__) #will get the root of tests, ie. mlops_individual_work/tests
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT) #will find the root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT,"data") #locates data path 