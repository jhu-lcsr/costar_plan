from costar_task_plan.abstract import AbstractCondition

import pybullet as pb

class CollisionCondition(AbstractCondition):
    def __init__(self, allowed):
        super(CollisionCondition, self).__init__()
        self.allowed = allowed

    def _check(self, world, state, actor=None, prev_state=None):
        pass
