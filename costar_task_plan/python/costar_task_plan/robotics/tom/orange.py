
from costar_task_plan.abstract import *

import numpy as np

class TomOrangesState(AbstractState):
    '''
    This state represents which orange we grasped and how we grasped it, plus whatever its state was (good or bad).
    '''
    def __init__(self, world):
        self.predicates = []
        self.world = world
        self.grasped_name = None # which orange did we grasp
        self.grasped_state = None # is this orange good or bad

class TomOrangesAction(AbstractAction):
    '''
    This action does basically nothing. It never acts.
    '''
    pass

class TomOrangesDynamics(AbstractDynamics):
    '''
    Check world state. If an actor closed its gripper near 
    '''
    def apply(self, state, action):
        for actor in state.world.actors:
            if actor.actor_type == 'robot':
                if actor.state.gripper_closed:
                    np.random

class TomOranges(AbstractActor):

    actor_type = 'env'

    '''
    This "actor" represents how a particular orange will change as it gets updated over time. We expect that it will mostly be updated by sensory data from the robot.
    '''
    def __init__(self, *args, **kwargs):
        super(TomOranges, self).__init__(dynamics=NullDynamics(), policy=NullPolicy())
