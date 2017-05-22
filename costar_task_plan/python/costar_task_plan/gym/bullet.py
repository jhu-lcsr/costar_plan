import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym.spaces import Box

from costar_task_plan.simulation import CostarBulletSimulation

import numpy as np

class BulletSimulationEnv(gym.Env, utils.EzPickle):

    def __init__(self, *args, **kwargs):
        self.client = CostarBulletSimulation(*args, **kwargs)
        self.action_space = self.client.robot.getActionSpace()

    def _step(self, action):
        '''
        Tick world with this action
        '''
        self.client.tick(action)
        return self.client.observe()

    def _reset(self):
        '''
        Reset client and world
        Return current features
        '''
        self.client.reset()
