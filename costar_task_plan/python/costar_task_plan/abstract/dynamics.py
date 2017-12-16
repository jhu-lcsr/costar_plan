
from world import *


class AbstractActor(object):

  actor_type = ''
  
  def __init__(self, state=None, policy=None, dynamics=None, features=None):
    self.policy = policy
    self.state = state
    self.dynamics = dynamics
    self.features = features
    self.last_state = None
    self.last_action = None
    self.event_info = None
    self.id = -1

  def setPolicy(self, policy):
    self.policy = policy

  def evaluate(self, world):
    if self.id < 0:
      raise Exception('Actor not initialized! Cannot evaluate.')

    # send the actor through to the policy
    a = self.policy(world, world.actors[self.id].state, self)
    return a

  # note: id determines which actor is which
  def setId(self, id):
    self.id = id

  def update(self, action, dt):
    self.last_state = self.state
    self.last_action = action
    self.state = self.dynamics(self.state, action, dt)
    return self.state

  def addReward(self, reward):
    self.reward.append(reward)

  def setFeatures(self, features):
    if self.policy is None:
      raise Exception('Features must be associated with a policy and this actor has no policy!')
    self.policy.setFeatures(features)

  def getFeatures(self):
    return self.policy.features

  def getState(self):
      raise NotImplementedError('state')

'''
Dynamics class governs how entities evolve over time
'''
class AbstractDynamics(object):

  def __init__(self,world):
    self.world = world

  def __call__(self, state, action, dt):
    assert(isinstance(state, AbstractState))
    assert(isinstance(action, AbstractAction))
    return self.apply(state, action, dt)

  def apply(self, state, action):
    raise Exception('dynamics.apply not implemented')

# AbstractPolicy
# Policy class governs what actions are chosen.
# It takes a world state plus some extra information (determining for example
# which actor should be taking the action, and what their predicted state will
# be).
class AbstractPolicy(object):

  def __init__(self, features=None):
    self.features = features

  def __call__(self, world, state, actor):
    assert(isinstance(state, AbstractState))
    assert(isinstance(world, AbstractWorld))
    return self.evaluate(world, state, actor)

  def setFeatures(self, features):
    self.features = features

  def evaluate(self, world, state, actor):
    raise Exception('policy.evaluate not implemented')

'''
Reward class governs what rewards we want to follow
'''
class AbstractReward(object):
  def __call__(self, world):
    return self.evaluate(world)

  def evaluate(self, world):
    raise Exception('reward.evaluate() not implemented')

class AbstractControllerPolicy(AbstractPolicy):
  action = None

  def setAction(self, action):
    if not isinstance(action, AbstractAction):
      raise TypeError('Must provide an action')
    self.action = action

  def evaluate(self, world, state):
    if self.action is None:
      return self._default()
    else:
      return self.action

class NullDynamics(AbstractDynamics):
  def apply(self, state, action):
      return None

class NullReward(AbstractReward):
    def __call__(self, world):
        return 0., 0.

