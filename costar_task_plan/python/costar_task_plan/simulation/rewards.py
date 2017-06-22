from costar_task_plan.abstract.dynamics import AbstractReward
from tf_conversions import posemath as pm

import numpy as np


class EuclideanDistanceReward(AbstractReward):
    def __init__(self, goal):
        self.goal = goal
    
    def __call__(self, world):
        robot = world.actors[0]
        T_ee = pm.fromMatrix(robot.robot.forward(robot.state.arm))
        T_obj = world.getObject(self.goal).state.T
        return np.linalg.norm(T_obj.p.tolist(), T_ee.p.tolist())
