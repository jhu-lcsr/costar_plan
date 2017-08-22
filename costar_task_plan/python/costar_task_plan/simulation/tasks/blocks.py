from abstract import AbstractTaskDefinition
from default import DefaultTaskDefinition
from costar_task_plan.simulation.world import *
from costar_task_plan.simulation.option import *
from costar_task_plan.simulation.reward import *
from costar_task_plan.simulation.condition import *
from costar_task_plan.abstract.simple_conditions import *
from costar_task_plan.abstract.task import *

import numpy as np
import os
import pybullet as pb
import rospkg


class BlocksTaskDefinition(DefaultTaskDefinition):

    '''
    Define a simple task. The robot needs to pick up and stack blocks of
    different colors in a particular order.
    '''

    # define object filenames
    block_urdf = "%s.urdf"
    model = "block"
    blocks = ["red", "blue", "yellow", "green"]

    # Objects are placed into a random stack.
    stack_pos = [
        # np.array([-0.5, 0., 0.]),
        np.array([-0.5, 0.1, 0.]),
        np.array([-0.5, 0.2, 0.]),
        np.array([-0.5, -0.1, 0.]),
        np.array([-0.5, -0.2, 0.]),
    ]

    over_final_stack_pos = np.array([-0.5, 0., 0.5])
    final_stack_pos = np.array([-0.5, 0., 0.05])
    grasp_q = (-0.27, 0.65, 0.65, 0.27)

    def __init__(self, stage, *args, **kwargs):
        '''
        Read in arguments defining how many blocks to create, where to create
        them, and the size of the blocks. Size is given as mean and covariance,
        blocks are placed at random.
        '''
        super(BlocksTaskDefinition, self).__init__(*args, **kwargs)
        self.stage = stage
        self.block_ids = []

    def _makeTask(self):

        tol = (0.005, 0.005)
        general_tol = (0.05, 0.025)

        # ====================================================================
        # First grasp -- pick up object from the side
        AlignOption = lambda goal: GoalDirectedMotionOption(
            self.world,
            goal,
            pose=((0.05, 0, 0.05), self.grasp_q),
            pose_tolerance=tol,
            joint_velocity_tolerance=0.05,)
        align_args = {
            "constructor": AlignOption,
            "args": ["block"],
            "remap": {"block": "goal"},
        }
        GraspOption = lambda goal: GoalDirectedMotionOption(
            self.world,
            goal,
            pose=((0.0, 0, 0.0), self.grasp_q),
            pose_tolerance=tol,
            joint_velocity_tolerance=0.05,)
        grasp_args = {
            "constructor": GraspOption,
            "args": ["block"],
            "remap": {"block": "goal"},
        }


        # ====================================================================
        # General actions -- lift the block back up again
        LiftOption = lambda: GeneralMotionOption(
            pose=(self.over_final_stack_pos, self.grasp_q),
            pose_tolerance=general_tol,
            joint_velocity_tolerance=0.05,)
        lift_args = {
            "constructor": LiftOption,
            "args": []
        }
        PlaceOption = lambda: GeneralMotionOption(
            pose=(self.final_stack_pos, self.grasp_q),
            pose_tolerance=tol,
            joint_velocity_tolerance=0.05,)
        place_args = {
            "constructor": PlaceOption,
            "args": []
        }
        close_gripper_args = {
            "constructor": lambda: CloseGripperOption(position=np.array([-0.6])),
            #"constructor": lambda: CloseGripperOption(),
            "args": []
        }
        open_gripper_args = {
            "constructor": OpenGripperOption,
            "args": []
        }

        if self.stage == 1:
            AlignStackOption = lambda goal: GoalDirectedMotionOption(
                self.world,
                goal,
                pose=((0.02, 0, 0.10), self.grasp_q),
                pose_tolerance=tol,
                joint_velocity_tolerance=0.05,)
            align_stack_args = {
                "constructor": AlignStackOption,
                "args": ["block"],
                "remap": {"block": "goal"},
            }
            StackOption = lambda goal: GoalDirectedMotionOption(
                self.world,
                goal,
                pose=((0.01, 0, 0.06), self.grasp_q),
                pose_tolerance=tol,
                joint_velocity_tolerance=0.05,
                closed_loop=True,)
            stack_args = {
                "constructor": StackOption,
                "args": ["block"],
                "remap": {"block": "goal"},
            }

            # ==================================================================== 
            # Pickup from somewhere
            pickup = TaskTemplate("pickup", None)
            pickup.add("align", None, align_args)
            pickup.add("grasp", "align", grasp_args)
            pickup.add("close_gripper", "grasp", close_gripper_args)
            pickup.add("lift", "close_gripper", lift_args)
            #task.add("place", "lift", place_args)

            # ==================================================================== 
            # Place on a stack
            place = TaskTemplate("place", "pickup")
            place.add("align_with_stack", "lift", align_stack_args)
            place.add("add_to_stack", "align_with_stack", stack_args)
            place.add("open_gripper", "add_to_stack", open_gripper_args)
            place.add("done", "open_gripper", lift_args)

            # ==================================================================== 
            # Create a task model
            pickup_args = {
                "task": pickup,
                "args": ["block"],
            }
            place_args = {
                "task": place,
                "args": ["block"],
            }
            task = Task()
            task.add("pickup", None, pickup_args)
            task.add("place", None, place_args)
        elif self.stage == 0:
            task = Task()
            task.add("align", None, align_args)
            task.add("grasp", "align", grasp_args)
            task.add("close_gripper", "grasp", close_gripper_args)
            task.add("lift", "close_gripper", lift_args)
            task.add("place", "lift", place_args)
            task.add("open_gripper", "place", open_gripper_args)
            task.add("done", "open_gripper", lift_args)

        return task

    def _addTower(self, pos, blocks, urdf_dir):
        '''
        Helper function that generats a tower containing listed blocks at the
        specific position
        '''
        z = 0.025
        ids = []
        for block in blocks:
            urdf_filename = os.path.join(
                urdf_dir, self.model, self.block_urdf % block)
            obj_id = pb.loadURDF(urdf_filename)
            r = self._sampleRotation()
            block_pos = self._samplePos(pos[0], pos[1], z)
            pb.resetBasePositionAndOrientation(
                obj_id,
                block_pos,
                r)
            self.addObject("block", "%s_block" % block, obj_id)
            z += 0.05
            ids.append(obj_id)
        return ids

    def _samplePos(self, x, y, z):
        diff = np.random.random((3,)) * [0.2, 0.1, 0.]
        return np.array([x,y,z]) + diff

    def _sampleRotation(self):
        '''
        Sample a random, small rotation.
        '''
        rpy = np.random.random((3,)) * 0.3
        rpy[0] = 0. # clear out the pitch
        r = kdl.Rotation.RPY(*list(rpy)).GetQuaternion()
        return r

    def _setup(self):
        '''
        Create task by adding objects to the scene
        '''

        rospack = rospkg.RosPack()
        path = rospack.get_path('costar_simulation')
        urdf_dir = os.path.join(path, self.urdf_dir)

        # placement =
        # np.random.randint(0,len(self.stack_pos),(len(self.blocks),))
        placement = np.array(range(len(self.stack_pos)))
        np.random.shuffle(placement)
        for i, pos in enumerate(self.stack_pos):
            blocks = []
            for idx, block in zip(placement, self.blocks):
                if idx == i:
                    blocks.append(block)
            ids = self._addTower(pos, blocks, urdf_dir)
            self.block_ids += ids

        self.world.addCondition(JointLimitViolationCondition(), -100,
                                "joints must stay in limits")
        self.world.addCondition(TimeCondition(15.), -100, "time limit reached")
        self.world.addCondition(AndCondition(
                                    ObjectIsBelowCondition("red_block", 0.55),
                                    ObjectIsBelowCondition("green_block", 0.55),
                                    ObjectIsBelowCondition("blue_block", 0.55),
                                    ObjectIsBelowCondition("yellow_block", 0.55),
                                ), -100, "block_too_high")
        self.world.reward = EuclideanReward("red_block")

        # =====================================================================
        # Set up the "first stage" of the tower -- so that we only need to
        # correctly place a single block.
        # NOTE: switching to give positive rewards for all to make it easier to
        # distinguish good training data from bad.
        if self.stage == 0:
            threshold = 0.035
            position_condition = AbsolutePositionCondition(
                self.over_final_stack_pos,
                self.grasp_q,
                0.05,
                0.025,
            )
            self.world.addCondition(
                    OrCondition(
                        ObjectAtPositionCondition("red_block",
                            self.final_stack_pos, threshold),
                        position_condition),
                    100,
                    "block in right position")
            self.world.addCondition(
                    OrCondition(
                        ObjectAtPositionCondition("blue_block",
                            self.final_stack_pos, threshold),
                        position_condition),
                    50,
                    "wrong block")
            self.world.addCondition(
                    OrCondition(
                        ObjectAtPositionCondition("green_block",
                            self.final_stack_pos, threshold),
                        position_condition),
                    50,
                    "wrong block")
            self.world.addCondition(
                    OrCondition(
                        ObjectAtPositionCondition("yellow_block",
                            self.final_stack_pos, threshold),
                        position_condition),
                    50,
                    "wrong block")

    def reset(self):
        '''
        Reset blocks to new random towers. Also resets the world and the
        configuration for all of the new objects, including the robot.
        '''

        # placement = np.random.randint(
        #        0,
        #        len(self.stack_pos),
        #        (len(self.blocks),))
        placement = np.array(range(len(self.stack_pos)))
        np.random.shuffle(placement)

        # loop over all stacks
        # pull out ids now associated with a stack
        for i, pos in enumerate(self.stack_pos):
            blocks = []
            for idx, block in zip(placement, self.block_ids):
                    if idx == i:
                        blocks.append(block)

            # add blocks to tower
            z = 0.025
            for block_id in blocks:
                block_pos = self._samplePos(pos[0], pos[1], z)
                r = self._sampleRotation()
                pb.resetBasePositionAndOrientation(
                    block_id,
                    block_pos,
                    r)
                z += 0.05

        self._setupRobot(self.robot.handle)

    def getName(self):
        return "blocks"
