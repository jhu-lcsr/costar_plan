from abstract import *
from node import *
from action import *

'''
Example class: do nothing.
'''
class NullInitialize(AbstractInitialize):
  def __call__(self, node):
    pass

'''
Create a set of children, each with an abstract policy associated.
'''
class PolicyInitialize(AbstractInitialize):
  def __init__(self, policies):
    self.policies = policies

  def __call__(self, node):
    for policy in self.policies:
      node.children.append(Node(action=MctsAction(policy=policy, ticks=self.ticks)))

'''
Create a set of children, each with an abstract policy associated.

Supposed to be used with a model trained via the Boltzman Q Policy. In
particular, based on the implementation in Keras RL:
https://github.com/matthiasplappert/keras-rl/blob/master/rl/policy.py#L94
'''
class LearnedPolicyInitialize(PolicyInitialize):
  def __init__(self,
      model, # neural net model
      weights_filename,
      tau=1.,
      clip=(-500., 500.),
      *args, **kwargs):

    super(LearnedPolicyInitialize,self).__init__(*args, **kwargs)

    self.model = model
    self.model.load_weights(weights_filename)
    self.tau = tau
    self.clip = clip

  def __call__(self, node):
    q_values = self.model.predict(np.array([node.world.initial_features]))[0]
    q_values = q_values.astype('float64')
    nb_actions = q_values.shape[0]
    exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
    probs = exp_values / np.sum(exp_values)

    for i, policy in enumerate(self.policies):
      node.children.append(Node(action=MctsAction(
        policy=policy,
        ticks=self.ticks,
        prior=probs[i])))

'''
Create a bunch of neural net policies from models and weights.
'''
class LateralMotionNNPolicyInitialize(PolicyInitialize):
  def __init__(self, models, weights_filenames):
    policies = []
    for model, filename in zip(models, weights_filenames):
      print 'Loading weights from file "%s..."'%filename
      policies.append(LateralMotionNeuralNetPolicy(model, filename))
    super(LateralMotionNNPolicyInitialize,self).__init__(policies)

'''
Steering angle:
Create a bunch of neural net policies from models and weights.
'''
class SteeringAngleNNPolicyInitialize(PolicyInitialize):
  def __init__(self, models, weights_filenames):
    policies = []
    for model, filename in zip(models, weights_filenames):
      print 'Loading weights from file "%s..."'%filename
      policies.append(SteeringAngleNeuralNetPolicy(model, filename))
    super(SteeringAngleNNPolicyInitialize,self).__init__(policies)


'''
Initialize based on a task model.
'''
class TaskModelInitialize(AbstractInitialize):
  def __init__(self, task):
    self.task = task

  def __call__(self, node):
    children = self.task.getChildren(node.tag)
    for child in children:
      option = self.task.getOption(child)
      policy, condition = option.makePolicy()
      node.children.append(Node(action=MctsAction(
        tag=child,
        policy=policy,
        condition=condition,),
        prior=1.0,))
