#!/usr/bin/env python

'''
By Chris Paxton
(c) 2017 Johns Hopkins University
See License for details
'''

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.initializations import *
from keras.regularizers import *

import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym.spaces import Discrete

import costar_task_plan.gym as ctpgym
from costar_task_plan.trainers import *

import numpy as np

try:
  from rl.agents import DDPGAgent, ContinuousDQNAgent
  from rl.agents.dqn import DQNAgent
  from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
  from rl.memory import SequentialMemory
  from rl.random import OrnsteinUhlenbeckProcess
except ImportError, e:
  print "This example uses keras-rl!"
  raise e

if __name__ == '__main__':
  np.random.seed(10)
  def getNeuralNet(env):
    actor = Sequential()
    actor.add(Dense(2,input_shape=(2,)))
    actor.add(Activation('relu'))
    actor.add(Dense(1))
    actor.add(Activation('linear'))
    return actor

  env = ctpgym.FunctionEnv()
  mu = getNeuralNet(env)
  sigma = getNeuralNet(env)
  V = getNeuralNet(env)

  np.random.seed(10001)
  trainer = ReinforceTrainer(env,
      actor=mu,
      std=sigma,
      critic=V,
      rollouts=10,
      steps=1000,
      reward_scale=1.,
      reward_baseline=0.,)
  trainer.compile(Adam(lr=.001, clipnorm=1e-8))
  trainer.train()

