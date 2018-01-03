from costar_task_plan.abstract import AbstractCondition

import numpy as np
import pybullet as pb
import PyKDL as kdl


class CollisionCondition(AbstractCondition):

    '''
    Basic condition. This fires if a given actor has collided with any entity
    that it is not allowed to collide with.
    '''

    def __init__(self, not_allowed):
        '''
        Takes in a list of objects that it is illegal to collide with. If any
        collisions are detected, then return False. Else return True.
        '''
        super(CollisionCondition, self).__init__()
        if not isinstance(not_allowed, int) \
                and not isinstance(not_allowed, long):

            raise TypeError('collision condition requires int handle')

        self.not_allowed = not_allowed

    def _check(self, world, state, actor, prev_state=None):
        # Get the pybullet handle for this actor
        handle = actor.robot.handle

        # Check for contact points
        pts = pb.getContactPoints(bodyA=handle, bodyB=self.not_allowed)

        # If return was empty then we can proceed
        return len(pts) == 0


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


def quaternion_distance(q1,q2):
    sumq2 = 0
    for i in range(4):
        sumq2 += q1[i] * q2[i]
    return 1 - sumq2

class GoalPositionCondition(AbstractCondition):

    '''
    True if the robot has not yet arrived at its goal position. The goal
    position is here defined as being within a certain distance of a point.
    '''

    def __init__(self, goal, pos, rot, pos_tol, rot_tol, v_tol=0.01):
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.v_tol = v_tol

        self.pos = pos
        self.rot = rot
        self.goal = goal

        pg = kdl.Vector(*self.pos)
        Rg = kdl.Rotation.Quaternion(*self.rot)
        self.T = kdl.Frame(Rg, pg)

    def _check(self, world, state, actor, prev_state=None):
        '''
        Returns true if within tolerance of position or any closer to goal
        object.
        '''

        # Get position of the object we are grasping. Since we compute a
        # KDL transform whenever we update the world's state, we can use
        # that for computing positions and stuff like that.
        obj = world.getObject(self.goal)
        T = obj.state.T * self.T

        dist = (state.T.p - T.p).Norm()
        arm_v = (prev_state.arm - state.arm) / world.dt
        #still_moving = np.any(np.abs(state.arm_v) > self.v_tol)
        still_moving = np.any(np.abs(arm_v) > self.v_tol)
        
        dq = quaternion_distance(
                state.T.M.GetQuaternion(),
                T.M.GetQuaternion())


        # print (self.T.p.Norm())
        # print (obj.state.T.p - T.p).Norm()
        # print T_robot.p, T.p, dist
        # print "> cond", dist, still_moving, state.arm_v, arm_v

        ###########Albert temporary code###########
        points = pb.getContactPoints(actor.robot.handle, obj.handle)
        if (points != []):
            return True and (dist > self.pos_tol or still_moving)
        ###########################################
        return dist > self.pos_tol or still_moving or dq > self.rot_tol

class AbsolutePositionCondition(AbstractCondition):

    '''
    True until the robot has gotten within some distance of a particular point
    in the world's coordinate frame.
    '''

    def __init__(self, pos, rot, pos_tol, rot_tol, v_tol=0.10):
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.v_tol = v_tol

        self.pos = pos
        self.rot = rot

        # If we are within this distance, the action failed.
        pg = kdl.Vector(*self.pos)
        Rg = kdl.Rotation.Quaternion(*self.rot)
        self.T = kdl.Frame(Rg, pg)

    def _check(self, world, state, actor, prev_state=None):
        '''
        Returns true until we are within tolerance of position
        '''
        T_robot = state.robot.fwd(state.arm)
        dist = (T_robot.p - self.T.p).Norm()
        arm_v = (prev_state.arm - state.arm) / world.dt
        still_moving = np.any(np.abs(arm_v) > self.v_tol)
        # still_moving = np.any(np.abs(state.arm_v) > self.v_tol)
        # print "cond2=", still_moving, dist > self.pos_tol,
        # print dist, ">", self.pos_tol

        return dist > self.pos_tol or still_moving

class ObjectIsBelowCondition(AbstractCondition):
    '''
    Make sure you don't move an object too high. This is particualrly to make
    sure that, in the stacking case, we don't attempt to place an object on top
    of itself during data generation...
    '''
    def __init__(self, obj, z):
        self.obj = obj
        self.z = z

    def _check(self, world, state, actor, prev_state=None):
        T = world.getObject(self.obj).state.T
        return T.p[2] < self.z

class ObjectAtPositionCondition(AbstractCondition):

    '''
    Check to see if a particular object is at some position
    '''

    def __init__(self, objname, pos, pos_tol, ):
        self.objname = objname
        self.pos_tol = pos_tol
        self.pos = pos

        # If we are within this distance, the action failed.
        self.p = kdl.Vector(*self.pos)

    def _check(self, world, *args, **kwargs):
        '''
        Returns true until we are within tolerance of position
        '''
        T = world.getObject(self.objname).state.T
        dist = (T.p - self.p).Norm()

        return dist > self.pos_tol

class AnyObjectAtRelativePositionCondition(AbstractCondition):
    '''
    This is set up to be the "goal" condition for the stacking task.
    It takes in two lists of objects, and will check to see if, for each
    object, there is another object (in the second set) that is at a particular
    position relative to it.
    '''

    def __init__(self, objs, bases, pos, rot, pos_tol, ):
        self.objs = objs
        self.bases = bases
        self.pos_tol = pos_tol
        self.pos = pos
        self.rot = rot

        # Compute the relative transformation from the "base" where we hope to
        # find the object (or, you know, where we're afraid to find it...)
        pg = kdl.Vector(*self.pos)
        Rg = kdl.Rotation.Quaternion(*self.rot)
        self.T = kdl.Frame(Rg, pg)

    def _check(self, world, *args, **kwargs):
        '''
        Loop over objs and bases, check to see which are in the right relative
        positions.
        '''
        for obj in self.objs:
            T_obj = world.getObject(obj).state.T
            for base in self.bases:
                T_base = world.getObject(base).state.T * self.T
                dist = (T_obj.p - T_base.p).Norm()
                if dist < self.pos_tol:
                    return False
        return True
                
        
class ObjectMovedCondition(AbstractCondition):
    '''
    Check to see if a particular 
    '''
    def __init__(self, objname, pos, pos_tol):
        self.objname = objname
        self.pos = pos
        self.p = kdl.Vector(*self.pos)
        self.pos_tol = pos_tol

    def _check(self, world, *args, **kwargs):
        '''
        Returns true until we are within tolerance of position
        '''
        T = world.getObject(self.objname).state.T
        #print T.p
        dist = (T.p - self.p).Norm()

        #print dist < self.pos_tol
        
        return dist < self.pos_tol
        
class GripperNotFullyClosedCondition(AbstractCondition):
    def __init__(self, threshold):
        self.threshold = threshold

    def _check(self, world, state, *args, **kwargs):
        # check gripper state
        pass

class GraspingObjectCondition(AbstractCondition):
    '''
    Check to see if the robot is grasping a particular object -- aka if the
    robot or any part of its gripper are in contact with an object.
    '''
    def __init__(self, objname):
        self.objname = objname

    def _check(self, world, *args, **kwargs):
        obj = world.getObject(self.objname)

        raise NotImplementedError('not yet implemented')

        return False
