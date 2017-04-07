
from dmp_policy import JointDmpPolicy, CartesianDmpPolicy

import numpy as np

from costar_task_plan.abstract import AbstractOption, AbstractCondition

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

  # Make a policy.
  def makePolicy(self,node):
    raise Exception('option.makePolicy not implemented!')
    return self.policy_type(
            goal_frame=self.goal_frame,
            dmp=self.instances[0],
            kinematics=self.kinematics)

  # Get the gating condition for a specific option.
  # - execution should continue until such time as this condition is true.
  def getGatingCondition(self, *args, **kwargs):
    return DmpCondition(
            goal_frame=self.goal_frame,
            dmp=self.instances[0],
            kinematics=self.kinematics,)

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

# This condition tells us whether or not we successfully arrived at the end of 
# an action. It is true while we should continue executing. If our ee pose is
# within tolerances and we are nearly stopped, it returns false.
class DmpCondition(AbstractCondition):

  def __init__(self, goal_frame, dmp, kinematics):
      self.goal_frame = goal_frame
      self.dmp = dmp
      self.kinematics = kinematics

  def __call__(self, world, state, actor=None, prev_state=None):
    if actor is None:
      actor = world.actors[0]
    
    return state.seq == 0 or np.all(state.q_dot < 1e-2)
