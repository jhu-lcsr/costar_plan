
from condition import GoalPositionCondition, AbsolutePositionCondition
from world import *

from costar_task_plan.abstract import AbstractOption
from costar_task_plan.abstract import AbstractCondition, TimeCondition
from costar_task_plan.abstract import AbstractPolicy

import numpy as np
import pybullet as pb
import PyKDL as kdl


class GoalDirectedMotionOption(AbstractOption):

    '''
    This represents a goal that will move us somewhere relative to a particular
    object, the "goal."

    This lets us sample policies that will take steps towards the goal. Actions
    are represented as a change in position; we'll assume that the controller
    running on the robot will be enough to take us to the goal position.
    '''

    def __init__(self, world, goal, pose, pose_tolerance=(1e-3, 1e-2),
            joint_velocity_tolerance=0.025, closed_loop=False, *args, **kwargs):

        self.goal = goal
        self.goal_id = world.getObjectId(goal)
        self.closed_loop = closed_loop
        if pose is not None:
            self.position, self.rotation = pose
            self.position_tolerance, self.rotation_tolerance = pose_tolerance
        else:
            raise RuntimeError('Must specify pose.')

    def makePolicy(self, world):
        # Make the policy.
        # Get the gating condition for a specific option.
        # - execution should continue until such time as this condition
        # is true.
        return CartesianMotionPolicy(self.position,
                                     self.rotation,
                                     goal=self.goal), \
            GoalPositionCondition(
                self.goal,  # what object we care about
                self.position,  # where we want to grab it
                self.rotation,
                            # rotation with which we want to grab it
                self.position_tolerance,
                self.rotation_tolerance)

    def samplePolicy(self, world):
        # Get the gating condition for a specific option.
        # - execution should continue until such time as this condition
        # is true.
        if not self.closed_loop:
            return CartesianMotionPolicy(self.position,
                                         self.rotation,
                                         goal=self.goal), \
                GoalPositionCondition(
                    self.goal,  # what object we care about
                    self.position,  # where we want to grab it
                    self.rotation,
                                # rotation with which we want to grab it
                    self.position_tolerance,
                    self.rotation_tolerance)
        else:
            obj = world.getObject(self.goal)
            pg = kdl.Vector(*self.position)
            Rg = kdl.Rotation.Quaternion(*self.rotation)
            Tg = kdl.Frame(Rg, pg)
            T = obj.state.T * Tg
            position = list(T.p)
            rotation = list(T.M.GetQuaternion())
            return CartesianMotionPolicy(position,
                                         rotation,
                                         goal=None), \
                    AbsolutePositionCondition(
                        position, # where we want to grab it
                        rotation, # rotation with which we want to grab it
                        self.position_tolerance,
                        self.rotation_tolerance,)

    def checkPrecondition(self, world, state):
        # Is it ok to begin this option?
        if not isinstance(world, AbstractWorld):
            raise RuntimeError(
                'option.checkPrecondition() requires a valid world!')
        if not isinstance(state, AbstractState):
            raise RuntimeError(
                'option.checkPrecondition() requires an initial state!')
        return True

    def checkPostcondition(self, world, state):
        # Did we successfully complete this option?
        if not isinstance(world, AbstractWorld):
            raise RuntimeError(
                'option.checkPostcondition() requires a valid world!')
        if not isinstance(state, AbstractState):
            raise RuntimeError(
                'option.checkPostcondition() requires an initial state!')
        return True


class GeneralMotionOption(AbstractOption):

    '''
    This motion is not parameterized by anything in particular. This lets us
    sample policies that will take us twoards this goal.
    '''

    def __init__(self, pose, pose_tolerance, joint_velocity_tolerance=0.025,
            *args, **kwargs):
        if pose is not None:
            self.position, self.rotation = pose
            self.position_tolerance, self.rotation_tolerance = pose_tolerance
        else:
            raise RuntimeError('must provide a position to move to.')

    def makePolicy(self, world):
        return CartesianMotionPolicy(self.position, self.rotation, goal=None)

    def samplePolicy(self, world):
        return CartesianMotionPolicy(self.position,
                                     self.rotation,
                                     goal=None), \
            AbsolutePositionCondition(
                self.position,  # where we want to grab it
                self.rotation,
                                # rotation with which we want to grab it
                self.position_tolerance,
                self.rotation_tolerance,
            )

    def checkPrecondition(self, world, state):
        # Is it ok to begin this option?
        if not isinstance(world, AbstractWorld):
            raise RuntimeError(
                'option.checkPrecondition() requires a valid world!')
        if not isinstance(state, AbstractState):
            raise RuntimeError(
                'option.checkPrecondition() requires an initial state!')
        return True

    def checkPostcondition(self, world, state):
        # Did we successfully complete this option?
        if not isinstance(world, AbstractWorld):
            raise RuntimeError(
                'option.checkPostcondition() requires a valid world!')
        if not isinstance(state, AbstractState):
            raise RuntimeError(
                'option.checkPostcondition() requires an initial state!')
        return True


class OpenGripperOption(AbstractOption):

    '''
    Look up the robot for the specific actor, and perform the appropriate "open
    gripper" action, as specified.

    This is something that we need to set from the particular robot, since each
    robot can have its own ways of closing a gripper and its own commands that
    are appropriate for this.
    '''

    def makePolicy(self, world):
        return OpenGripperPolicy(), TimeCondition(world.time() + 1.0)

    def samplePolicy(self, world):
        return OpenGripperPolicy(), TimeCondition(world.time() + 1.0)

    def checkPrecondition(self, world, state):
        return True

    def checkPostcondition(self, world, state):
        return True


class CloseGripperOption(AbstractOption):

    '''
    As above, this option creates the approprite policies for closing a gripper,
    and does nothing else. These policies count on certain information
    associated with the actor's state in order to function.
    '''
    def __init__(self, position=None):
        '''
        Parameters:
        ----------
        position: Position to send when closing the gripper. Defaults to
                  "none", which uses the position set per robot.
        '''
        self.position = position

    def makePolicy(self, world):
        return CloseGripperPolicy(pos=self.position), \
               TimeCondition(world.time() + 3.0)

    def samplePolicy(self, world):
        return CloseGripperPolicy(pos=self.position), \
               TimeCondition(world.time() + 3.0)

    def checkPrecondition(self, world, state):
        return True

    def checkPostcondition(self, world, state):
        return True


class CartesianMotionPolicy(AbstractPolicy):

    def __init__(self, pos, rot, goal=None, cartesian_vel=0.5, angular_vel=0.5):
        self.pos = pos
        self.rot = rot
        self.goal = goal
        self.cartesian_vel = cartesian_vel
        self.angular_vel = angular_vel

        pg = kdl.Vector(*self.pos)
        Rg = kdl.Rotation.Quaternion(*self.rot)
        self.T = kdl.Frame(Rg, pg)

    def evaluate(self, world, state, actor):
        '''
        Compute IK to goal pose for actor.
        Goal pose is computed based on the position of a goal object, if one
        has been specified; otherwise we assume the goal has been specified
        in global coordinates.
        '''

        if self.goal is not None:
            # Get position of the object we are grasping. Since we compute a
            # KDL transform whenever we update the world's state, we can use
            # that for computing positions and stuff like that.
            obj = world.getObject(self.goal)
            T = obj.state.T * self.T
        else:
            # We can just use the cached position, since this is a known world
            # position and not something special.
            T = self.T

        if actor.robot.grasp_idx is None:
            raise RuntimeError(
                'Did you properly set up the robot URDF to specify grasp frame?')

        # =====================================================================
        # Compute transformation from current to goal frame
        T_r_goal = state.T.Inverse() * T

        # Interpolate in position alone
        dist = T_r_goal.p.Norm()
        step = min(self.cartesian_vel*world.dt, dist)
        p = T_r_goal.p / dist * step

        # Interpolate in rotation alone
        angle, axis = T_r_goal.M.GetRotAngle()

        angle = min(self.angular_vel*world.dt, angle)
        R = kdl.Rotation.Rot(axis, angle)
        T_step = state.T * kdl.Frame(R, p)

        # =====================================================================
        # Issue computing inverse kinematics
        # compos, comorn, ifpos, iforn, lwpos, lworn = pb.getLinkState(actor.robot.handle, actor.robot.grasp_idx)
        # print lwpos, lworn
        # q = pb.calculateInverseKinematics(actor.robot.handle,
        #                                  actor.robot.grasp_idx,
        #                                  targetPosition=position,
        #                                  targetOrientation=rotation)
        # from tf_conversions import posemath as pm
        # mat = pm.toMatrix(T)
        # print mat
        # print actor.robot.kinematics.forward(state.arm)

        # =====================================================================
        # Compute motion goak and send
        q_goal = actor.robot.ik(T_step, state.arm)
        # print q_goal, state.arm, state.arm_v
        return SimulationRobotAction(arm_cmd=q_goal)


class OpenGripperPolicy(AbstractPolicy):

    '''
    This simple policy just looks at robot internals to send the appropriate
    "open gripper" command.
    '''

    def evaluate(self, world, state, actor):
        return SimulationRobotAction(gripper_cmd=state.robot.gripperOpenCommand())


class CloseGripperPolicy(AbstractPolicy):

    '''
    This simple policy just looks at robot internals to send the appropriate
    "close gripper" command.
    '''

    def __init__(self,pos=None):
        self.pos = pos
        self.step = -0.1

    def evaluate(self, world, state, actor):
        if self.pos is None:
            pos = state.robot.gripperCloseCommand()
        else:
            pos = self.pos
        pos_cmd = max(state.gripper + self.step, pos)
        return SimulationRobotAction(gripper_cmd=pos_cmd)
