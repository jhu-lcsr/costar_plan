#!/usr/bin/env python

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

import task_tree_search.gym as ttsgym
from task_tree_search.trainers import *

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

np.random.seed(10)
def getNeuralNet(env):
  actor = Sequential()
  actor.add(Dense(2,input_shape=(2,)))
  actor.add(Activation('relu'))
  actor.add(Dense(1))
  actor.add(Activation('linear'))
  return actor

env = ttsgym.FunctionEnv()
mu = getNeuralNet(env)
sigma = getNeuralNet(env)

random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.15, size=1)
trainer = ReinforceTrainer(env,
    actor=mu,
    stddev=sigma,
    rollouts=10,
    steps=1000,
    reward_scale=1.,
    reward_baseline=0.,)
trainer.compile(Adam(lr=.001, clipnorm=1e-8))

np.random.seed(10001)
trainer.train()
