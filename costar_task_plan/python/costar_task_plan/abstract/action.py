import numpy as np

class AbstractAction(object):
  '''
  Abstract version of the action.
  '''

  def __init__(self):
      self.code = None
      self.error = None

  def toArray(self):
    '''
    toArray(): convert this action to a numpy array
    '''
    raise Exception('conversion to array not implemented!')

  def dist(self, other_action):
    '''
    compute distance metric to an action
    '''
    return np.linalg.norm(self.toArray() - other_action.toArray())

  def getDescription(cls):
    '''
    You may want to override this for things with more complex action spaces.
    '''
    return "action"
