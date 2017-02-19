# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from costar_task_plan.ros.core import CostarWorld
from costar_task_plan.ros.core import TomWorld


'''
This takes a task definition and sets up an action server.
'''
class PlanningServer(object):

    WORLD_T = CostarWorld

    def __init__(self, world_t=WORLD_T):
        self.world_t = world_t

    def parse(self, cmd):
        pass

'''
This subset of the class instantiates a very particular version of the CoSTAR
planning world.
'''
class TomPlanningServer(ob):
    WORLD_T = TomWorld

    def __init__(self):
        super(TomPlanningServer, self).__init__(self.WORLD_T)
