
from condition import GoalPositionCondition
from world import *

from costar_task_plan.abstract import AbstractOption
from costar_task_plan.abstract import AbstractCondition
from costar_task_plan.abstract import AbstractPolicy

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

    def __init__(self, world, goal, pose, pose_tolerance=(1e-2,1e-2), *args, **kwargs):
        self.goal = goal
        self.goal_id = world.getObjectId(goal)
        if pose is not None:
            self.position, self.rotation = pose
            self.position_tolerance, self.rotation_tolerance = pose_tolerance
        else:
            raise RuntimeError('Must specify pose.')

    def makePolicy(self, world):
        return CartesianMotionPolicy(self.position, self.rotation)

    def samplePolicy(self, world):
        return CartesianMotionPolicy(self.position,
                self.rotation,
                goal=self.goal)

    def getGatingCondition(self, *args, **kwargs):
        # Get the gating condition for a specific option.
        # - execution should continue until such time as this condition is true.
        return GoalPositionCondition(self.position,
                self.rotation, 
                self.position_tolerance,
                self.rotation_tolerance)
        
    def checkPrecondition(self, world, state):
        # Is it ok to begin this option?
        if not isinstance(world, AbstractWorld):
            raise RuntimeError(
                'option.checkPrecondition() requires a valid world!')
        if not isinstance(state, AbstractState):
            raise RuntimeError(
                'option.checkPrecondition() requires an initial state!')
        raise NotImplementedError(
            'option.checkPrecondition() not yet implemented!')

    def checkPostcondition(self, world, state):
        # Did we successfully complete this option?
        if not isinstance(world, AbstractWorld):
            raise RuntimeError(
                'option.checkPostcondition() requires a valid world!')
        if not isinstance(state, AbstractState):
            raise RuntimeError(
                'option.checkPostcondition() requires an initial state!')
        raise NotImplementedError(
            'option.checkPostcondition() not yet implemented!')


class GeneralMotionOption(AbstractOption):
    '''
    This motion is not parameterized by anything in particular. This lets us 
    sample policies that will take us twoards this goal. 
    '''
    def __init__(self, pose, *args, **kwargs):
        if pose is not None:
            self.position, self.rotation = pose


class CartesianMotionPolicy(AbstractPolicy):
    def __init__(self, pos, rot, goal=None):
        self.pos = pos
        self.rot = rot
        self.goal = goal

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

        compos, comorn, ifpos, iforn, lwpos, lworn = pb.getLinkState(actor.robot.handle, actor.robot.grasp_idx)
        print lwpos, lworn, "arm=",state.arm
        if self.goal is not None:
            # Get position of the object we are grasping
            obj = world.getObject(self.goal)
            pos = obj.state.base_pos
            rot = obj.state.base_rot

            p = kdl.Vector(*pos)
            R = kdl.Rotation.Quaternion(*lworn)
            T = kdl.Frame(R,p) * self.T
            #position = [T.p[0], T.p[1], T.p[2]]
            #rotation = T.M.GetQuaternion()
        else:
            #position = self.pos
            #rotation = self.rot
            T = self.T

        if actor.robot.grasp_idx is None:
            raise RuntimeError('Did you properly set up the robot URDF to specify grasp frame?')

        # Issue computing inverse kinematics
        #compos, comorn, ifpos, iforn, lwpos, lworn = pb.getLinkState(actor.robot.handle, actor.robot.grasp_idx)
        #print lwpos, lworn
        #q = pb.calculateInverseKinematics(actor.robot.handle,
        #                                  actor.robot.grasp_idx,
        #                                  targetPosition=position,
        #                                  targetOrientation=rotation)
        cmd = actor.robot.ik(T)
        print "q =",cmd
        return SimulationRobotAction(cmd=cmd)
