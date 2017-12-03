
from world import *
from condition import *

'''
Option
An option is a sub-policy that will satisfy some intermediate goal.

The Option class also encloses code to specifically train options for various
problems. The important functions here are:
  - makeTrainingWorld(): create a world configured to train this option.
  - makePolicy(): get a policy that will follow this option.
'''
class AbstractOption(object):

  def __init__(self, name=""):
    self.name = name
    self.preconditions = []
    self.postconditions = []

  @property
  def get_name(self):
    return name

  def makeWorld(self, *args, **kwargs):
    '''
    Create a world for testing this specific option
    '''
    raise Exception('cannot make training world for this option')

  def makePolicy(self, world, *args, **kwargs):
      '''
      Get policy for performing this option.
      Get the gating condition for a specific option.
      - execution should continue until such time as this condition is true.
      '''
      raise Exception('option.makePolicy not implemented!')

  def samplePolicy(self, world, *args, **kwargs):
      '''
      Generate a randomized version of a policy from some distribution.

      Get policy for performing this option.
      Get the gating condition for a specific option.
      - execution should continue until such time as this condition is true.
      '''
      return makePolicy(world, *args, **kwargs)

  def addPrecondition(self, condition):
      self.preconditions.append(condition)
    
  def addPostCondition(self, condition):
      self.postconditions.append(condition)

  def checkPrecondition(self, world, state):
    '''
    Is it ok to begin this option?
    '''
    if not isinstance(world, AbstractWorld):
        raise RuntimeError('option.checkPrecondition() requires a valid world!')
    if not isinstance(state, AbstractState):
        raise RuntimeError('option.checkPrecondition() requires an initial state!')
    for c in self.preconditions:
        if not c(world, state):
            return False
    return True

  def checkPostcondition(self, world, state):
    '''
    Did we successfully complete this option?
    '''
    if not isinstance(world, AbstractWorld):
        raise RuntimeError('option.checkPostcondition() requires a valid world!')
    if not isinstance(state, AbstractState):
        raise RuntimeError('option.checkPostcondition() requires an initial state!')
    for c in self.postconditions:
        if not c(world, state):
            return False
    return True

class NullOption(AbstractOption):
  '''
  Create an empty option for the root of the tree. It's always complete, and
  will therefore return an empty policy and an empty gating condition.
  '''

  def __init__(self):
    super(NullOption, self).__init__(name="root")

  @property
  def get_name(self):
    return name

  def makeWorld(self, *args, **kwargs):
    raise Exception('cannot make training world for this option')

  def makePolicy(self, world):
    return None, None

  def checkPrecondition(self, world, state):
    return True

  def checkPostcondition(self, world, state):
    return True
