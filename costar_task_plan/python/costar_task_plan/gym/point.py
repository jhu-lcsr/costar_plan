import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from gym.spaces import Box

import numpy as np


'''
Toy environment.
'''
class PointEnv(gym.Env):

  def __init__(self, center):
    self.observation_space = Box(np.array([-1.,-1.]),np.array([1.,1.]))
    self.action_space = Box(np.array([-1.,-1.]),np.array([1.,1.]))
    self.center = center
  
  def __del__(self):
    pass

  def _render(self, mode='human', close=False):
    pass

  def _step(self, action):

    r = -np.linalg.norm(self.center-action)

    return action, r, True, {}

  def _reset(self):
    return [0.,0.]

  def _configure_environment(self):
    pass
