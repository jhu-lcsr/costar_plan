
from costar_task_plan.abstract import AbstractPolicy

# This class takes the planning scene interface from the world and uses
# it to compute (and follow!) a DMP. It will check the current time vs. its
# tick rate before recomputing the DMP.
class DmpPolicy(AbstractPolicy):

    def __init__(self, features=None, dmp=None):
      raise NotImplementedError('DMP policy not set up!')

    def evaluate(self, world, state, actor=None):
      raise NotImplementedError('DMP policy not set up!')
        
