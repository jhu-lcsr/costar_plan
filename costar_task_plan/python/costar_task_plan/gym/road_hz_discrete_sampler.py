import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym.spaces import Discrete

import matplotlib.pyplot as plt

import task_tree_search.road_world as rw
import task_tree_search.road_world.planning as rwp
import task_tree_search.abstract as abstract
import task_tree_search.mcts as mcts

from utils import *
from road_hz_option import *

'''
Discrete sampling "option" environment. In this case, we take a 
'''
class RoadWorldDiscreteSamplerEnv(RoadWorldOptionEnv, utils.EzPickle):
  metadata = {'render.modes': ['human']}

  def __init__(self,
      verbose=False,
      speed_limit=5,
      randomize=True,
      planning=False,
      lateral=False,
      ltl=False,
      option="Default",
      sampler=None,
      max_depth=10,
      *args,
      **kwargs):

    super(RoadWorldDiscreteSamplerEnv, self).__init__(
        verbose=verbose,
        speed_limit=speed_limit,
        randomize=randomize,
        planning=planning,
        option=option,
        lateral=lateral,
        ltl=ltl,
        *args,
        **kwargs)

    self.depth = 0
    self.max_depth = max_depth
    self.lateral = lateral
    self.ltl = ltl
    self._fig = None
    self._last_action = None

    minf, maxf = self._world.getFeatureBounds()
    self._f_shape = minf.shape
    self._last_f = np.zeros(self._f_shape)

    if sampler is None:
      if self.lateral:
        self._sampler = rwp.LateralMotionPolicySample()
      else:
        self._sampler = rwp.SteeringAnglePolicySample()
    else:
      self._sampler = sampler
    self.action_space = Discrete(self._sampler.numOptions())
    self.observation_space = Box(
        np.append(np.append(minf, np.zeros((self.action_space.n,))), minf),
        np.append(np.append(maxf, np.zeros((self.action_space.n,))), maxf))
    self._reset()


  # update function: set the actor's policy from the random control
  # or from whatever else we got as an input
  def _step(self, action):
    A = self._sampler.getOption(self._node, action)
    self._node.children.append(A.apply(self._node))
    self.depth += 1

    # set current state to the child of the last action
    self._node = self._node.children[-1]
    R = self._node.reward
    terminal = self._node.terminal
    F1 = self._node.features()
    F_action = np.zeros(self.action_space.n,)
    F_action[action] = 1.0
    F = np.append(np.append(self._last_f, F_action), F1)
    self._last_f = F1

    if self.depth >= self.max_depth:
      terminal = True
    #elif terminal and self.depth < self.max_depth:
    #  # repeat last state
    #  R *= (self.max_depth - self.depth)

    return (F, R, terminal, {"action":action})

  def _reset(self):
    self.depth = 0
    self._configure_environment()
    self._root = mcts.Node(world=self._world)
    self._world.updateFeatures()
    self._last_f = self._world.computeFeatures()
    self._node = self._root
    prev_f = np.append(np.zeros(self._f_shape), np.zeros(self.action_space.n,))
    return np.append(prev_f, self._last_f)

  def _render(self, mode='human', close=False):
    if mode is 'human':
      if close and self._fig is not None:
        plt.close(self._fig)
      else:
        if self._fig is None:
          self._fig = plt.figure()
          self._fig.show()
        rwp.drawTree(self._root,blocking=False,fig=self._fig)
