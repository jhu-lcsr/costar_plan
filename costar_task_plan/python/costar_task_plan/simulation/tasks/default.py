
from abstract import AbstractTaskDefinition

from costar_task_plan.simulation.camera import Camera

import numpy as np
import os
import pybullet as pb


class DefaultTaskDefinition(AbstractTaskDefinition):

    # These are for the UR5
    joint_positions = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]

    # define folder for blocks
    urdf_dir = "urdf"
    sdf_dir = "sdf"
    model_file_name = "model.sdf"

    def __init__(self, *args, **kwargs):
        super(DefaultTaskDefinition, self).__init__(*args, **kwargs)
        self.objs = []
        self.addCamera(
            Camera("right", [-0.25, 0., 0.25], distance=2.0, roll=0., pitch=np.pi / 2, yaw=0.4))

    def _setupRobot(self, handle):
        self.robot.place([0, 0, 0], [0, 0, 0, 1], self.joint_positions)
        self.robot.arm(self.joint_positions, pb.POSITION_CONTROL)
        self.robot.gripper(0, pb.POSITION_CONTROL)

    def reset(self):
        '''
        Basic reset needs to reconfigure the world state -- set things back to
        the way they should be.
        '''
        self.world.done = False
        self.world.ticks = 0.
        for obj in self.objs:
            pb.removeBody(obj)
        self._setup()
        self._setupRobot(self.robot.handle)
