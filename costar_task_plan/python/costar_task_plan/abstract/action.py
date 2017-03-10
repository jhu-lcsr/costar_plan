import numpy as np

class AbstractAction(object):

  '''
  toArray(): convert this action to a numpy array
  '''
  def toArray(self):
    raise Exception('conversion to array not implemented!')

  '''
  compute distance metric to an action
  '''
  def dist(self, other_action):
    return np.linalg.norm(self.toArray() - other_action.toArray())

