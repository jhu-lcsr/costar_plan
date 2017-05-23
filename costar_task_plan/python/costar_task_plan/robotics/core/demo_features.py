from costar_task_plan.abstract import AbstractFeatures
from costar_task_plan.robotics.representation import RobotFeatures

import tf_conversions.posemath as pm

class DemoFeatures(AbstractFeatures):
    '''
    Wrap old feature computation code
    '''

    def __init__(self,kdl_kin,config):
        self.config = config
        self.kdl_kin = kdl_kin
        self.features = RobotFeatures(self.config, self.kdl_kin)

    def compute(self, world, state):
        '''
        Compute LfD features
        '''
        if state.reference is not None:
            ee = pm.fromMatrix(self.kdl_kin.forward(state.q))
            if state.gripper_closed:
                gripper = 1.
            else:
                gripper = 0.
            f = self.features.GetFeatures(
                    ee,
                    state.seq / len(state.traj),
                    world.observation,
                    [state.reference.goal],
                    gripper)
            return f
        else:
            return None

    def updateBounds(self, world):
        raise Exception('feature.updateBounds not yet implemented!')

    def getBounds(self):
        raise Exception('feature.getBounds not yet implemented!')
