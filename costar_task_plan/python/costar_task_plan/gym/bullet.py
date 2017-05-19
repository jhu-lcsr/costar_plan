import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym.spaces import Box

import numpy as np

class BulletSimulationEnv(gym.Env, utils.EzPickle):

    def __init__(self, *args, **kwargs):
        self.client = CostarBulletSimulation(*args, **kwargs)

    def _step(self, action):
        '''
        Tick world with this action
        '''
        pass

    def _reset(self):
        '''
        Reset client and world
        Return current features
        '''
        self.client.reset()
