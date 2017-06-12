from abstract import AbstractTaskDefinition
from costar_task_plan.simulation.world import *
from costar_task_plan.simulation.option import *

import numpy as np
import os
import pybullet as pb
import rospkg


class BlocksTaskDefinition(AbstractTaskDefinition):

    '''
    Define a simple task. The robot needs to pick up and stack blocks of
    different colors in a particular order.
    '''

    joint_positions = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]

    # define folder for blocks
    urdf_dir = "urdf"

    # define object filenames
    block_urdf = "%s.urdf"
    model = "block"
    blocks = ["red", "blue", "yellow", "green"]

    # Objects are placed into a random stack.
    stack_pos = [
            np.array([-0.5, 0., 0.]),
            np.array([-0.5, 0.2, 0.]),
            np.array([-0.5, -0.2, 0.]),
            ]

    def __init__(self, *args, **kwargs):
        '''
        Read in arguments defining how many blocks to create, where to create
        them, and the size of the blocks. Size is given as mean and covariance,
        blocks are placed at random.
        '''
        super(BlocksTaskDefinition, self).__init__(*args, **kwargs)

    def _makeTask(self):
        GraspOption = lambda goal: GoalDirectedMotionOption(self.world, goal, (None, None))
        grasp_args = {
                "constructor": GraspOption,
                "args": ["block"],
                "remap": {"block": "goal"},
                }
        LiftOption = lambda: GeneralMotionOption(None)
        lift_args = {
                "constructor": LiftOption,
                "args": []
                }
        PlaceOption = lambda goal: GoalDirectedMotionOption
        grasp_args = {
                "constructor": PlaceOption,
                "args": ["block"],
                "remap": {"block": "goal"},
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

    def _addTower(self, pos, blocks, urdf_dir):
        '''
        Helper function that generats a tower containing listed blocks at the
        specific position
        '''
        z = 0.025
        for block in blocks:
            urdf_filename = os.path.join(urdf_dir, self.model, self.block_urdf%block)
            obj_id = pb.loadURDF(urdf_filename)
            pb.resetBasePositionAndOrientation(
                    obj_id,
                    (pos[0], pos[1], z),
                    (0,0,0,1))
            self.addObject("block", obj_id)
            z += 0.05

    def _setup(self):
        '''
        Create task by adding objects to the scene
        '''

        rospack = rospkg.RosPack()
        path = rospack.get_path('costar_objects')
        urdf_dir = os.path.join(path, self.urdf_dir)

        placement = np.random.randint(0,len(self.stack_pos),(len(self.blocks),))
        for i, pos in enumerate(self.stack_pos):
            blocks = []
            for idx, block in zip(placement, self.blocks):
                if idx == i:
                    blocks.append(block)
            self._addTower(pos, blocks, urdf_dir)

    def _setupRobot(self, handle):
        self.robot.place([0,0,0],[0,0,0,1],self.joint_positions)
        self.robot.arm(self.joint_positions, pb.POSITION_CONTROL)
        self.robot.gripper(0, pb.POSITION_CONTROL)

    def reset(self):
        pass

    def getName(self):
        return "blocks"
