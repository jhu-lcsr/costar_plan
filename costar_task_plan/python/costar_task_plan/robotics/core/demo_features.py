from costar_task_plan.abstract import AbstractFeatures

import tf_conversions.posemath as pm

class DemoFeatures(AbstractFeatures):
    '''
    Wrap old feature computation code
    '''

    def __init__(self,config,kdl_kin):
        self.config = config
        self.kdl_kin = kdl_kin
        self.features = RobotFeatures(self.config, self.kdl_kin)

    def compute(self, world, state):
        if state.reference is not None:
            ee = pm.fromMatrix(self.kdl_kin.forward(state.q))
            obj = world.observation[state.reference.goal]
        
