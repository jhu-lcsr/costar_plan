from costar_task_plan.abstract import AbstractReward

from tf_conversions import posemath as pm

import numpy as np


class EuclideanReward(AbstractReward):

    '''
    Distance based purely on euclidean distance to object
    '''

    def __init__(self, goal):
        '''
        Goal is an object to approach
        '''
        self.goal = goal

    def evaluate(self, world):
        '''
        Reward is 0 at object.
        '''
        robot_actor = world.actors[0]
        T_ee = pm.fromMatrix(robot_actor.robot.fwd(robot_actor.state.arm))
        T_obj = world.getObject(self.goal).state.T
        return -(T_obj.p - T_ee.p).Norm(), 0
