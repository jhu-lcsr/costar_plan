from .util import *
from .service_caller import ServiceCaller

class StackManager(ServiceCaller):
    '''
    This class creates and holds the different services we need to call to
    create the stacking task and execute it.
    '''
    objs = ["red_cube", "green_cube", "blue_cube", "yellow_cube"]

    def __init__(self):
        self.detect = GetDetectObjectsService()
        self.grasp = GetSmartGraspService()
        self.place = GetSmartPlaceService()
        self.reset()
        self.reqs = {}
        self.children = {}

    def reset(self):
        self.done = False
        self.current = None

    def addRequest(self, parents, name, srv, req):
        self.reqs[name] = (srv, req)
        if not isinstance(parents, list):
            parents = [parents]
        for parent in parents:
            if parent not in self.children:
                self.children[parent] = []
            self.children[parent].append(name)

    def tick(self):
        if self.running:
            self.done = False
        elif self.current in self.children:
            # This one has a child to execute
            self.done = False
        else:
            self.done = True
        print self.children
        print self.children[self.current]

