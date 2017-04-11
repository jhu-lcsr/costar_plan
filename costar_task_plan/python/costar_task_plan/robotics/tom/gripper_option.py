
from costar_task_plan.abstract import AbstractOption, AbstractPolicy

# Send a gripper command: either open or close
class TomGripperOption(AbstractOption):
  def __init__(self, close):
    self.close_gripper = close

# Send an open command and wait until that's successful
class TomGripperOpenOption(TomGripperOption):
  def __init__(self):
    super(TomGripperOpenOption,self).__init__(False)

  def makePolicy(self):
    return TomCloseGripperPolicy()

# Send a close command and wait until that's successful
class TomGripperCloseOption(TomGripperOption):
  def __init__(self):
    super(TomGripperCloseOption,self).__init__(True)

  def makePolicy(self):
    return TomOpenGripperPolicy()

class TomCloseGripperPolicy(AbstractPolicy):
    def __call__(self, *args, **kwargs):
        return CostarAction(close_gripper=True)

class TomOpenGripperPolicy(AbstractPolicy):
    def __call__(self, *args, **kwargs):
        return CostarAction(open_gripper=True)

# Determines whether or not a gripper has been set to the appropriate state.
# This is based on the world information.
class TomGripperCondition(AbstractCondition):

  def __call__(self, world, state, actor=None, prev_state=None):
