from .util import *

class StackManager(object):
    '''
    This class creates and holds the different services we need to call to
    create the stacking task and execute it.
    '''
    def __init__(self):
        self.detect = GetDetectObjectsService()
        self.grasp = GetSmartGraspService()
        self.place = GetSmartPlaceService()

