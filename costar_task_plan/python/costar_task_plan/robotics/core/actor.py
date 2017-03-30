
import numpy as np

from costar_task_plan.abstract import *

# State of a particular actor. It's the joint state, nice and simple.
class CostarState(AbstractState):
  def __init__(self, world, q=np.array([]), dq=np.array([])):
    self.predicates = []
    self.q = q
    self.dq = dq

    # This should (hopefully) be a reference the world. it hardly matters
    # for something like this, though -- our states hold very little
    # information.
    self.world = world

  def toArray(self):
    return self.q

# Actions for a particular actor. This is very simple, and just represents a
# joint motion, normalized over some period of time.
class CostarAction(AbstractAction):
  def __init__(self, dq=np.array([])):
    self.dq = dq

  def toArray(self):
    return self.dq

# This actor represents a robot in the world.
# It's mostly defined by its config -- most of the actual logic that uses this
# is defined in the world's hook() function that gets called after every
# update.
class CostarActor(AbstractActor):

  def __init__(self, config, *args, **kwargs):
    super(CostarActor, self).__init__(*args, **kwargs)
    self.config = config
    self.joints = config['joints']
    self.dof = self.config['dof']
    if not self.dof == len(self.joints):
      raise RuntimeError('You configured the robot joints wrong')


