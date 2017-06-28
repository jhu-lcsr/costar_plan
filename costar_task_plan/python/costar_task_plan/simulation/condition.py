from costar_task_plan.abstract import AbstractCondition

import numpy as np
import pybullet as pb
import PyKDL as kdl

class CollisionCondition(AbstractCondition):
    '''
    Basic condition. This fires if a given actor has collided with any entity
    that it is not allowed to collide with.
    '''
    def __init__(self, allowed):
        super(CollisionCondition, self).__init__()
        self.allowed = allowed

    def _check(self, world, state, actor, prev_state=None):
        pass

class JointLimitViolationCondition(AbstractCondition):
    '''
    True if all arm positions are within joint limits as defined by the
    robot's kinematics.
    '''
    def __init__(self):
        pass

    def _check(self, world, state, actor, prev_state=None):
        '''
        Use KDL kinematics to determine if the joint limits were acceptable
        '''
        #print state.arm
        return actor.robot.kinematics.joints_in_limits(state.arm).all()

class SafeJointLimitViolationCondition(AbstractCondition):
    '''
    True if all arm positions are within joint limits as defined by the
    robot's kinematics.
    '''
    def __init__(self):
        pass

    def _check(self, world, state, actor, prev_state=None):
        '''
        Use KDL kinematics to determine if the joint limits were acceptable
        '''
        return actor.robot.kinematics.joints_in_safe_limits(state.arm).all()

class GoalPositionCondition(AbstractCondition):
    '''
    True if the robot has not yet arrived at its goal position. The goal
    position is here defined as being within a certain distance of a point.
    '''
    def __init__(self, goal, pos, rot, pos_tol, rot_tol):
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol

        self.pos = pos
        self.rot = rot
        self.goal = goal
        
        # If we are within this distance, the action failed.
        self.dist = np.linalg.norm(self.pos)

        pg = kdl.Vector(*self.pos)
        Rg = kdl.Rotation.Quaternion(*self.rot)
        self.T = kdl.Frame(Rg, pg)

    def _check(self, world, state, actor, prev_state=None):
        '''
        Returns true if within tolerance of position or any closer to goal
        object.
        '''
        return True


class AbsolutePositionCondition(AbstractCondition):
    '''
    True until the robot has gotten within some distance of a particular point
    in the world's coordinate frame.
    '''
    def __init__(self, goal, pos, rot, pos_tol, rot_tol):
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol

        self.pos = pos
        self.rot = rot
        self.goal = goal
        
        # If we are within this distance, the action failed.
        self.dist = np.linalg.norm(self.pos)

        pg = kdl.Vector(*self.pos)
        Rg = kdl.Rotation.Quaternion(*self.rot)
        self.T = kdl.Frame(Rg, pg)

    def _check(self, world, state, actor, prev_state=None):
        '''
        Returns true until we are within tolerance of position
        '''
        return True
