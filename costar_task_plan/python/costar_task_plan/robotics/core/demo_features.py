from costar_task_plan.abstract import AbstractFeatures
from costar_task_plan.robotics.representation import RobotFeatures

import numpy as np
import tf_conversions.posemath as pm

from numba import jit


class DemoFeatures(AbstractFeatures):

    '''
    Wrap old feature computation code
    '''

    def __init__(self, kdl_kin, config):
        self.config = config
        self.kdl_kin = kdl_kin
        self.features = RobotFeatures(self.config, self.kdl_kin)

    def compute(self, world, state):
        '''
        Compute LfD features
        '''
        print "asdf"
        if state.reference is not None:
            ee = pm.fromMatrix(self.kdl_kin.forward(state.q))
            if state.gripper_closed:
                gripper = 1.
            else:
                gripper = 0.
            f = np.array(self.features.GetFeatures(
                ee,
                state.seq / len(state.traj.points),
                world.observation,
                ['time', state.reference.goal],
                gripper))
            return f
        else:
            print "!!!! DemoFeatures " 
            return None

    def updateBounds(self, world):
        raise Exception('feature.updateBounds not yet implemented!')

    def getBounds(self):
        raise Exception('feature.getBounds not yet implemented!')

@jit(nopython=True)
def GetFeatures(ee, world, objs, size):
    f = np.zeros((size,))
    return f

@jit(nopython=True)
def GetFeaturesForTrajectory(ee, world, objs, size):
    f = np.zeros((len(ees),size))
    return f
