
from costar_task_plan.abstract import Condition

class InGateCondition(AbstractCondition):
  name = "in gate"

  def __init__(self, gate):
    self.gate = gate

  def _check(self, world):
    return self.gate.Contains(world.getLearner().state)

  def _name(self):
    return name
