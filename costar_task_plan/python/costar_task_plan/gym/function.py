import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from utils import *

import pygame as pg
import costar_task_plan.road_world as rw
import costar_task_plan.abstract as abstract
import costar_task_plan.draw as draw

from gym.spaces import Box

import numpy as np

def identity(x):
  return x

'''
Toy environment. This one will plot out some kind of function as a simple toy
example for reinforcement learning.

Our job is to take X steps to follow this function.
'''
class FunctionEnv(gym.Env, utils.EzPickle):

  def __init__(self, f=identity):
    self.f = f
    self.x = 0
    self.t = 0
    self.observation_space = Box(np.array([-100]),np.array([100]))
    self.action_space = Box(np.array([-1]),np.array([1]))
  
  def __del__(self):
    pass

  def _render(self, mode='human', close=False):
    pass

  def _step(self, action):
    self.x += action[0]
    self.t += 1

    y = self.f(self.x)
    y_true = self.f(self.t)

    dy = y - y_true
    r = (10 - abs(dy))/10

    return [self.x, self.t], r, self.t >= 10, {}

  def _reset(self):
    self.x = 0
    self.t = 0
    return [0, 0]

  def _configure_environment(self):
    self.x = 0
    self.t = 0
    return [0, 0]
