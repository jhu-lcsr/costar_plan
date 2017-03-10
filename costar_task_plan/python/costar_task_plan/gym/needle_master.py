import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from gym.spaces import Box

import numpy as np


'''
Needle Master environment (placeholder only).
'''
class NeedleMasterEnv(gym.Env, utils.EzPickle):

  def __init__(self):
    self.observation_space = Box(np.array([-1,-1]),np.array([1,1]))
    self.action_space = Box(np.array([-1,-1]),np.array([1,1]))
    pass
  
  def __del__(self):
    pass

  def _render(self, mode='human', close=False):
    pass

  def _step(self, action):
    return None

  def _reset(self):
    return None

  def _configure_environment(self):
    pass

