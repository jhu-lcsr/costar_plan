
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
        self.grasped = False
        self.orange_is_good = None

class TomOrangesAction(AbstractAction):
    '''
    This action does basically nothing. It never acts.
    '''
    pass

class TomOrangesExpectedDynamics(AbstractDynamics):

    def __init__(self, p_grasp_success = 0.75, p_good_orange = 0.5):
        self.p_grasp_success = p_grasp_success
        self.p_good_orange = p_good_orange

    '''
    Check world state. If an actor closed its gripper near 
    '''
    def apply(self, state, action):
        for actor in state.world.actors:
            if actor.actor_type == 'robot':
                # grasping logic
                if actor.state.gripper_closed and state.grasped == False:
                    if np.random.random() < self.p_grasp_success:
                        self.grasped = True
                elif actor.state.gripper_status == 'open' and state.grasped == True:
                    self.grasped = False
                
                # testing logic

class TomOranges(AbstractActor):

    actor_type = 'env'

    '''
    This "actor" represents how a particular orange will change as it gets updated over time. We expect that it will mostly be updated by sensory data from the robot.
    '''
    def __init__(self, *args, **kwargs):
        super(TomOranges, self).__init__(dynamics=NullDynamics(), policy=NullPolicy())
