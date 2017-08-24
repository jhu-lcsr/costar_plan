
from abstract import AbstractTaskDefinition

from costar_task_plan.simulation.camera import Camera

import numpy as np
import os
import pybullet as pb


class DefaultTaskDefinition(AbstractTaskDefinition):

    # These are for the UR5
    joint_positions = np.array([0.30, -1.33, -1.80, -0.27, 1.50, 1.60])
    random_limit = 0.8

    # define folder for blocks
    urdf_dir = "urdf"
    sdf_dir = "sdf"
    model_file_name = "model.sdf"

    def __init__(self, *args, **kwargs):
        super(DefaultTaskDefinition, self).__init__(*args, **kwargs)
        self.objs = []
        self.addCamera(
            #Camera("right", [-0.5, 0., 0.15], distance=0.7, roll=0.0,
            Camera("right", [-0.45, 0., 0.25], distance=0.7, roll=0.0,
                image_width=64,
                image_height=64,
                pitch=-45,
                yaw=-90,
                fov=40))

    def _setupRobot(self, handle):
        q = self.joint_positions + \
            (self.random_limit*np.random.random(self.joint_positions.shape) - 
             self.random_limit*0.5)
        self.robot.place([0, 0, 0], [0, 0, 0, 1], q)
        self.robot.arm(q, pb.POSITION_CONTROL)
        self.robot.gripper(self.robot.gripperOpenCommand(), pb.POSITION_CONTROL)

    def reset(self):
        '''
        Basic reset needs to reconfigure the world state -- set things back to
        the way they should be.
        '''
        for obj in self.objs:
            pb.removeBody(obj)
        self._setup()
        self._setupRobot(self.robot.handle)
