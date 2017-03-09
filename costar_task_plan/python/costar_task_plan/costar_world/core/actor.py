
from costar_task_plan.abstract import *

'''
State of a particular actor.
'''
class CostarState(AbstractState):
  def __init__(self, world, q=np.array([])):
    self.predicates = []
    self.q = q
    self.world = world

  def toArray(self):
    return self.q

'''
Actions for a particular actor.
'''
class CostarAction(AbstractAction):
  def __init__(self, dq=np.array([])):
    self.dq = dq

  def toArray(self):
    return self.dq

'''
This actor represents a robot in the world.
'''
class CostarActor(AbstractActor):

  def __init__(self, config, *args, **kwargs):
    super(CostarActor, self).__init__(*args, **kwargs)
    self.config = config
    self.dof = self.config['dof']

  def addArm(*args, **kwargs):
    pass

  def addBase(*args, **kwargs):
    pass


