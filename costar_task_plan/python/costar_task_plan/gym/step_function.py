import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from utils import *

import pygame as pg
import costar_task_search.road_world as rw
import costar_task_search.abstract as abstract
import costar_task_search.draw as draw

from gym.spaces import Box, Discrete

import numpy as np

def identity(x):
  return x

'''
Toy environment. This one will plot out some kind of function as a simple toy
example for reinforcement learning.

This is a DISCRETE version of the problem.

Our job is to take X steps to follow this function.
'''
class StepFunctionEnv(gym.Env, utils.EzPickle):

  def __init__(self, f=identity):
    self.f = f
    self.x = 0
    self.t = 0
    self.observation_space = Box(np.array([-1,-1]),np.array([1,1]))
    self.action_space = Discrete(3)
  
  def __del__(self):
    pass

  def _render(self, mode='human', close=False):
    pass

  def _step(self, action):
    if action == 0:
      dx = -1
    elif action == 1:
      dx = 1
    else:
      dx = 0
    
    self.x += dx
    self.t += 1

    y = np.floor(self.f(self.x))
    y_true = np.floor(self.f(self.t))

    dy = (y_true - y) / 10.
    r = 1 - (dy*dy)

    return [self.x, self.t], r, self.t >= 10, {}

  def _reset(self):
    self.x = 0
    self.t = 0
    return [0, 0]

  def _configure_environment(self):
    self.x = 0
    self.t = 0
    return [0, 0]
