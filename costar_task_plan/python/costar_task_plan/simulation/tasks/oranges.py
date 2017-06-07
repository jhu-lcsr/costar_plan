from abstract import AbstractTaskDefinition
from costar_task_plan.simulation.world import *
from costar_task_plan.simulation.option import *

import numpy as np
import os
import pybullet as pb
import rospkg


class SortingTaskDefinition(AbstractTaskDefinition):
    '''
    Define the simple sorting task.
    '''

    joint_positions = [0.30, -0.5, -1.80, -0.27, 1.50, 1.60]

    tray_dir = "tray"
    tray_urdf = "traybox.urdf"

    spawn_pos_min = np.array([-0.4 ,-0.25, 0.10])
    spawn_pos_max = np.array([-0.65, 0.25, 0.155])
    spawn_pos_delta = spawn_pos_max - spawn_pos_min

    tray_poses = [np.array([-0.5, 0., 0.0]),
                  np.array([0., +0.6, 0.0]),
                  np.array([-1.0, -0.6, 0.0])]

    def __init__(self, robot, *args, **kwargs):
        '''
        Your desription here
        '''
        super(SortingTaskDefinition, self).__init__(robot, *args, **kwargs)

    def _makeTask(self):
        '''
        Create the high-level task definition used for data generation.
        '''
        GraspOption = lambda goal: GoalDirectedMotionOption
        grasp_args = {
                "constructor": GraspOption,
                "args": ["red"],
                "remap": {"red": "goal"},
                }
        LiftOption = lambda: GeneralMotionOption
        lift_args = {
                "constructor": LiftOption,
                "args": []
                }
        wait_args = {
                "constructor": GeneralMotionOption,
                "args": []
                }
        place_args = {
                "constructor": GeneralMotionOption,
                "args": []
                }
        close_gripper_args = {
                "constructor": GeneralMotionOption,
                "args": []
                }
        open_gripper_args = {
                "constructor": GeneralMotionOption,
                "args": []
                }

        # Create a task model
        task = Task()
        task.add("grasp", None, grasp_args)
        task.add("close_gripper", "grasp", close_gripper_args)
        task.add("lift", "close_gripper", grasp_args)
        task.add("place", "lift", grasp_args)
        task.add("open_gripper", "place", open_gripper_args)

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
            pb.resetBasePositionAndOrientation(obj_id, position, (0,0,0,1))

    def reset(self):
        for obj_id, position in zip(self.trays, self.tray_poses):
            pb.resetBasePositionAndOrientation(obj_id, position, (0,0,0,1))
        self.robot.place([0,0,0],[0,0,0,1],self.joint_positions)

    def getName(self):
        return "oranges"
