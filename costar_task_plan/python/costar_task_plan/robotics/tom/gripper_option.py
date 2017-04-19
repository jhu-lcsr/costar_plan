
import numpy as np

from costar_task_plan.abstract import AbstractOption
from costar_task_plan.abstract import AbstractPolicy
from costar_task_plan.abstract import AbstractCondition
from costar_task_plan.robotics.core import CostarAction

# Send a gripper command: either open or close
class TomGripperOption(AbstractOption):
  def __init__(self, close):
    self.close_gripper = close

# Send an open command and wait until that's successful
class TomGripperOpenOption(TomGripperOption):
  def __init__(self):
    super(TomGripperOpenOption,self).__init__(False)

  #Create a policy that will send a command to close the gripper
  def makePolicy(self):
    return TomCloseGripperPolicy()

  # Holds until the world reports that TOM's gripper is closed
  def getGatingCondition(self, *args, **kwargs):
    return TomGripperIsClosedCondition()

# Send a close command and wait until that's successful
class TomGripperCloseOption(TomGripperOption):
  def __init__(self):
    super(TomGripperCloseOption,self).__init__(True)

  def makePolicy(self):
    return TomCloseGripperPolicy()

  # Holds until the world reports TOM's gripper is open
  def getGatingCondition(self, *args, **kwargs):
    return TomGripperNotClosedCondition()

# =============================================================================
# POLICIES

# Close the gripper
class TomCloseGripperPolicy(AbstractPolicy):
  def evaluate(self, world, state, *args, **kwargs):
    print "======================"
    print "close"
    print "======================"
    return CostarAction(q=state.q,
                        dq=np.zeros(state.q.shape),
                        reference=None,
                        gripper_cmd="close")

# Open the gripper
class TomOpenGripperPolicy(AbstractPolicy):
  def evaluate(self, world, state, *args, **kwargs):
    print "======================"
    print "open"
    print "======================"
    return CostarAction(q=state.q,
                        dq=np.zeros(state.q.shape),
                        reference=None,
                        gripper_cmd="open")

# =============================================================================
# Conditions

# Determines whether or not a gripper has been set to the appropriate state.
# This is based on the world information.
class TomGripperCondition(AbstractCondition):

  def __init__(self, gripper_closed):
      self.gripper_closed = gripper_closed

  def __call__(self, world, state, actor=None, prev_state=None):
    return state.gripper_closed == self.gripper_closed

# Hodls while the gripper is open or closing
class TomGripperNotClosedCondition(TomGripperCondition):
    def __init__(self):
        super(TomGripperNotClosedCondition, self).__init__(False)

# Holds true while the gripper is closed or opening
class TomGripperIsClosedCondition(TomGripperCondition):
    def __init__(self):
        super(TomGripperIsClosedCondition, self).__init__(True)
