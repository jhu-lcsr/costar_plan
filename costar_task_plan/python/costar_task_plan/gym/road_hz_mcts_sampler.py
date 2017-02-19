import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym.spaces import Discrete

import matplotlib.pyplot as plt
import operator

import task_tree_search.road_world as rw
import task_tree_search.road_world.planning as rwp
import task_tree_search.abstract as abstract
import task_tree_search.mcts as mcts

from utils import *
from road_hz_option import *

'''
Discrete sampling "option" environment. In this case, we take a 
'''
class RoadWorldMctsSamplerEnv(RoadWorldOptionEnv, utils.EzPickle):
  metadata = {'render.modes': ['human']}

  def __init__(self, verbose=False, speed_limit=5,
      randomize=True,
      planning=False,
      option="Default",
      sampler=None,
      lateral=True,
      ltl=False,
      max_iter=20,
      max_depth=5,):

    super(RoadWorldMctsSamplerEnv, self).__init__(verbose, speed_limit, randomize, planning, option)
    self._fig = None
    self._reset()
    if sampler is None:
      if self.lateral:
        self._sampler = rwp.LateralMotionPolicySample()
      else:
        self._sampler = rwp.SteeringAnglePolicySample()
    else:
      self._sampler = sampler
    self.policies = mcts.DefaultMctsPolicies(sample=self._sampler)
    self.action_space = Discrete(self._sampler.numOptions())
    self.max_iter = max_iter
    self.max_depth = max_depth

  # update function: set the actor's policy from the random control
  # or from whatever else we got as an input
  def _step(self, action):

    A = self._sampler.getOption(self._node, action)
    new_node = A.apply(self._node)
    self._node.children.append(new_node)
    self.policies.select(self._root, self.max_depth, False)
    self._iter += 1

    # try to add a new node to the tree
    self._node = self.policies.getNext(self._root, self.max_depth)
    while self._node is None:
      # keep simulating play until we get there
      self.policies.select(self._root, self.max_depth, False)
      self._node = self.policies.getNext(self._root, self.max_depth)

    if self._iter >= self.max_iter:
      node = self._root 
      R = 0
      while node.n_visits > 0:
        R += node.reward
        if len(node.children) > 0:
          reward = [(i, child.avg_reward) for i, child in enumerate(node.children)]
          child_idx, _ = max(reward, key=operator.itemgetter(1))
          node = node.children[child_idx]
        else:
          break
      terminal = True
    else:
      # set current state to the child of the last action
      #self._node = self._node.children[-1]
      R = 0;
      terminal = False

    F1 = self._node.features()
    return (F1, R, terminal, {})

  def _reset(self):
    self._configure_environment()
    self._root = mcts.Node(world=self._world)
    self._node = self._root
    self._iter = 0

    self._world.updateFeatures()
    return self._world.initial_features

  def _render(self, mode='human', close=False):
    if mode is 'human':
      if close and self._fig is not None:
        plt.close(self._fig)
      else:
        if self._fig is None:
          self._fig = plt.figure()
          self._fig.show()
        rwp.drawTree(self._root,blocking=False,fig=self._fig)
