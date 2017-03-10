
from abstract_world import *
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

  def __init__(self, name="", goal_conditions=[], failure_conditions=[]):
    self.goal_conditions = goal_conditions
    self.failure_conditions = failure_conditions
    self.name = name

  @property
  def get_name(self):
    return name

  '''
  Create a world for testing this specific option
  '''
  def makeWorld(self, *args, **kwargs):
    raise Exception('cannot make training world for this option')

  def makePolicy(self):
    raise Exception('option.makePolicy not implemented!')

  '''
  Get the gating condition for a specific option.
  - execution should continue until such time as this condition is true.
  '''
  def getGatingCondition(self, state, *args, **kwargs):
    if not isinstance(state, AbstractState):
        raise RuntimeError('option.getGatingCondition() requires an initial state!')
    raise NotImplementedError('option.getGatingCondition() not yet implemented!')

  '''
  Is it ok to begin this option?
  '''
  def checkPrecondition(self, world, state):
    if not isinstance(world, AbstractWorld):
        raise RuntimeError('option.checkPrecondition() requires a valid world!')
    if not isinstance(state, AbstractState):
        raise RuntimeError('option.checkPrecondition() requires an initial state!')
    raise NotImplementedError('option.checkPrecondition() not yet implemented!')

  '''
  Did we successfully complete this option?
  '''
  def checkPostcondition(self, world, state):
    if not isinstance(world, AbstractWorld):
        raise RuntimeError('option.checkPostcondition() requires a valid world!')
    if not isinstance(state, AbstractState):
        raise RuntimeError('option.checkPostcondition() requires an initial state!')
    raise NotImplementedError('option.checkPostcondition() not yet implemented!')


