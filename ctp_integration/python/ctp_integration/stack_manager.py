from .util import *

class StackManager(object):
    '''
    This class creates and holds the different services we need to call to
    create the stacking task and execute it.
    '''
    self.objs =
        objs = ["red_cube", "green_cube", "blue_cube", "yellow_cube"]
    def __init__(self):
        self.detect = GetDetectObjectsService()
        self.grasp = GetSmartGraspService()
        self.place = GetSmartPlaceService()

    def sample(self):
        pass

    def test(self):
        pass
