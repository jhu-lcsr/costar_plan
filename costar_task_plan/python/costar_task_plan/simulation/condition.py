from costar_task_plan.abstract import AbstractCondition

import pybullet as pb

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

class GoalPositionCondition(AbstractCondition):
    '''
    True if the robot has not yet arrived at its goal position. The goal
    position is here defined as being within a certain distance of a point.
    '''
    def __init__(self, pos, rot, pos_tol, rot_tol):
        self.pos = pos
        self.rot = rot
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol

    def _check(self, world, state, actor, prev_state=None):
        return True
