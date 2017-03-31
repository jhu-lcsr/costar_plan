
from costar_task_plan.abstract import AbstractOption

# Send a gripper command: either open or close
class TomGripperOption(AbstractOption):
  def __init__(self, close):
    self.close_gripper = close

# Send an open command and wait until that's successful
class TomGripperOpenOption(TomGripperOption):
  def __init__(self):
    super(TomGripperOpenOption,self).__init___(False)

# Send a close command and wait until that's successful
class TomGripperCloseOption(TomGripperOption):
  def __init__(self):
    super(TomGripperCloseOption,self).__init___(True)

