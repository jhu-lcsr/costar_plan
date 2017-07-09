import numpy as np

class AbstractState(object):

  def __init__(self):
    self.predicates = []
    self.event_info = None

  '''
  toArray(): convert this state to a numpy array
  '''
  def toArray(self):
    raise Exception('conversion to array not implemented!')

  def toParams(self, action):
    '''
    Extract the set of parameters from an action according to the current state
    of the world/actor.
    '''
    raise Exception('conversion of action to array not implemented!') 

  def updatePredicates(self, world, actor):
    self.predicates = [check(world, self, actor, actor.last_state)
        for (name, check)
        in world.predicates]

  '''
  compute distance metric to a state
  '''
  def dist(self, other_state):
    return np.linalg.norm(self.toArray() - other_state.toArray())


