
from dmp_policy import JointDmpPolicy, CartesianDmpPolicy

from costar_task_plan.abstract import AbstractOption

class DmpOption(AbstractOption):

  def __init__(self, policy_type, kinematics, goal_frame, model, instances=[], attached_frame=None):
    if isinstance(policy_type, str):
      # parse into appropriate constructor
      if policy_type == 'joint':
        policy_type = JointDmpPolicy
        raise NotImplementedError('Joint space skills not currently implemented.')
      elif policy_type == 'cartesian':
        policy_type = CartesianDmpPolicy
      else:
        raise RuntimeError('invalid option for DMP policy type: %s'%policy_type)
    if not isinstance(policy_type, type):
      raise RuntimeError('invalid data type for DMP policy')
    if attached_frame is not None:
      raise NotImplementedError('attached frame is not yet supported')

    self.policy_type = policy_type
    self.kinematics = kinematics
    self.instances = instances
    self.model = model
    self.attached_frame = attached_frame

  def makePolicy(self,):
    raise Exception('option.makePolicy not implemented!')

  # Get the gating condition for a specific option.
  # - execution should continue until such time as this condition is true.
  def getGatingCondition(self, state, *args, **kwargs):
    if not isinstance(state, AbstractState):
        raise RuntimeError('option.getGatingCondition() requires an initial state!')
    raise NotImplementedError('option.getGatingCondition() not yet implemented!')

  # Is it ok to begin this option?
  def checkPrecondition(self, world, state):
    if not isinstance(world, AbstractWorld):
        raise RuntimeError('option.checkPrecondition() requires a valid world!')
    if not isinstance(state, AbstractState):
        raise RuntimeError('option.checkPrecondition() requires an initial state!')
    raise NotImplementedError('option.checkPrecondition() not yet implemented!')

  # Did we successfully complete this option?
  def checkPostcondition(self, world, state):
    if not isinstance(world, AbstractWorld):
        raise RuntimeError('option.checkPostcondition() requires a valid world!')
    if not isinstance(state, AbstractState):
        raise RuntimeError('option.checkPostcondition() requires an initial state!')
    raise NotImplementedError('option.checkPostcondition() not yet implemented!')
