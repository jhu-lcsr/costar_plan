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
def getNeuralNet(env,outputs):
  model = Sequential()
  model.add(Dense(8,input_shape=(2,)))
  model.add(Activation('relu'))
  model.add(Dense(8))
  model.add(Activation('relu'))
  model.add(Dense(outputs))
  model.add(Activation('linear'))
  return model

env = ttsgym.StepFunctionEnv()
actor = getNeuralNet(env,3)
critic = getNeuralNet(env,1)

trainer = DiscreteReinforceTrainer(env, actor, critic,
    rollouts=100,
    learning_rate=1e-2,)
trainer.compile()

np.random.seed(10001)
trainer.train()
