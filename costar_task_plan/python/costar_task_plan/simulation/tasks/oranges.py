from abstract import AbstractTaskDefinition
from costar_task_plan.simulation.world import *
from costar_task_plan.simulation.option import *

import numpy as np
import os
import pybullet as pb
import rospkg


class OrangesTaskDefinition(AbstractTaskDefinition):

    '''
    Define the simple sorting task.
    '''

    joint_positions = [0.30, -0.5, -1.80, -0.27, 1.50, 1.60]

    urdf_dir = "urdf"
    tray_dir = "tray"
    tray_urdf = "traybox.urdf"

    spawn_pos_min = np.array([-0.4, -0.25, 0.10])
    spawn_pos_max = np.array([-0.65, 0.25, 0.155])
    spawn_pos_delta = spawn_pos_max - spawn_pos_min

    tray_poses = [np.array([-0.5, 0., 0.0]),
                  np.array([0., +0.6, 0.0]),
                  np.array([-1.0, -0.6, 0.0])]

    def __init__(self, robot, *args, **kwargs):
        '''
        Your desription here
        '''
        super(OrangesTaskDefinition, self).__init__(robot, *args, **kwargs)

    def _makeTask(self):
        '''
        Create the high-level task definition used for data generation.
        '''
        task = Task()
        return task

    def _setup(self):
        '''
        Create the mug at a random position on the ground, handle facing
        roughly towards the robot. Robot's job is to grab and lift.
        '''

        rospack = rospkg.RosPack()
        path = rospack.get_path('costar_objects')
        urdf_dir = os.path.join(path, self.urdf_dir)
        tray_filename = os.path.join(urdf_dir, self.tray_dir, self.tray_urdf)

        for position in self.tray_poses:
            obj_id = pb.loadURDF(tray_filename)
            pb.resetBasePositionAndOrientation(obj_id, position, (0, 0, 0, 1))

    def _setupRobot(self, handle):
        '''
        Properly place and configure the robot.
        '''
        self.robot.place([0, 0, 0], [0, 0, 0, 1], self.joint_positions)
        if self.robot.arm_name == "ur5":
            self.robot.arm(self.joint_positions, pb.POSITION_CONTROL)
        elif self.robot.arm_name == "iiwa":
            raise NotImplementedError('iiwa')
        else:
            raise NotImplementedError(
                'whatever you entered: "%s"' % self.robot.arm_name)

        if self.robot.gripper_name == "robotiq_2_finger":
            self.robot.gripper(0, pb.POSITION_CONTROL)
        else:
            raise NotImplementedError(
                'whatever you entered: "%s"' % self.robot.gripper_name)

    def reset(self):
        for obj_id, position in zip(self.trays, self.tray_poses):
            pb.resetBasePositionAndOrientation(obj_id, position, (0, 0, 0, 1))
        self.robot.place([0, 0, 0], [0, 0, 0, 1], self.joint_positions)

    def getName(self):
        return "oranges"
