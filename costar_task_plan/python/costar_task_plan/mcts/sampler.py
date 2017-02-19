from abstract import *
from action import *

import numpy as np
import operator

'''
Policy that represents only a single policy
'''
class SinglePolicySample(AbstractSample):
  def __init__(self, policy, ticks=10):
    self.policy = policy
    self.ticks = ticks

  def numOptions(self):
    return 1

  def getOption(self, node, idx):
    return MctsAction(policy=self.policy, id=0, ticks=self.ticks)

  def _sample(self, node):
    return MctsAction(policy=self.policy, id=0, ticks=self.ticks)

  def getPolicies(self, node):
    return [self.policy]

  def getName(self):
    return "single"

'''
Empty sampler for fixed-width trees
'''
class NullSample(object):
  def _sample(self, node):
    return None
  def getOption(self, node, idx):
    return None
  def numOptions(self):
    return 0
  def getPolicies(self):
    return []
  def getName(self):
    return "null"

'''
Take a normal sampler and put it in a weird order
'''
class LearnedOrderPolicySample(AbstractSample):
  def __init__(self, model, weights_filename, sampler):
    self.model = model
    self.model.load_weights(weights_filename)
    self.sampler = sampler

  def _sample(self, node, *args, **kwargs):
    idx = len(node.children)
    A = self.model.predict(np.array([node.world.initial_features]))[0]
    A = [a[0] for a in sorted(enumerate(A),key=operator.itemgetter(1))]
    if idx < len(A):
      return self.sampler.getOption(node, A[idx-1])
    else:
      return None

  def getName(self):
    return "learned"+self.sampler.getName()
