
import argparse
import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.initializations import *
from keras.regularizers import *

from task_tree_search.road_world.config import *

def normal_init(shape, name=None):
  return normal(shape, scale=1e-4, name=name)

def cdqn_get_v_model(env):
  V_model = Sequential()
  V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
  V_model.add(Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  V_model.add(Activation('relu'))
  V_model.add(Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  V_model.add(Activation('relu'))
  V_model.add(Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  V_model.add(Activation('relu'))
  V_model.add(Dense(1))
  V_model.add(Activation('linear'))
  return V_model

def cdqn_get_mu_model(env, nb_actions):
  mu_model = Sequential()
  mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
  mu_model.add(Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  mu_model.add(Activation('relu'))
  mu_model.add(Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  mu_model.add(Activation('relu'))
  mu_model.add(Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  mu_model.add(Activation('relu'))
  mu_model.add(Dense(nb_actions))
  mu_model.add(Activation('tanh'))
  return mu_model

def cdqn_get_L_model(env, nb_actions):
  action_input = Input(shape=(nb_actions,), name='action_input')
  observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
  x = merge([action_input, Flatten()(observation_input)], mode='concat')
  x = Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG))(x)
  x = Activation('relu')(x)
  x = Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG))(x)
  x = Activation('relu')(x)
  x = Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG))(x)
  x = Activation('relu')(x)
  x = Dense(((nb_actions * nb_actions + nb_actions) / 2))(x)
  x = Activation('linear')(x)
  L_model = Model(input=[action_input, observation_input], output=x)
  return L_model

def dqn_get_actor(env, nb_actions):
  actor = Sequential()
  actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
  actor.add(Dense(TASK_FF_LAYER_SIZE,
    W_regularizer=l2(OPTION_W_REG_L1),
    activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(TASK_FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(TASK_FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(nb_actions))
  actor.add(Activation('linear'))
  return actor

def dqn_get_validation_actor(env, nb_actions):
  actor = Sequential()
  actor.add(Dense(TASK_FF_LAYER_SIZE,
    input_shape=env.observation_space.shape,
    W_regularizer=l2(OPTION_W_REG_L1),
    activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(TASK_FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(TASK_FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(nb_actions))
  actor.add(Activation('linear'))
  return actor

def dqn_get_lstm_actor(env, nb_actions, window_length):
  actor = Sequential()
  actor.add(TimeDistributed(Flatten(input_shape=(1,window_length,) + env.observation_space.shape)))
  actor.add(TimeDistributed(
    Dense(FF_LAYER_SIZE,
      W_regularizer=l2(OPTION_W_REG),
      activity_regularizer=activity_l2(OPTION_A_REG))))
  actor.add(TimeDistributed(Activation('relu')))
  actor.add(LSTM(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(nb_actions))
  actor.add(Activation('linear'))
  return actor

def dqn_get_lstm_validation_actor(env, nb_actions, window_length):
  actor = Sequential()
  actor.add(TimeDistributed(
    Dense(FF_LAYER_SIZE,
      input_shape=(window_length,) + env.observation_space.shape,
      W_regularizer=l2(OPTION_W_REG),
      activity_regularizer=activity_l2(OPTION_A_REG))))
  actor.add(Activation('relu'))
  actor.add(Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(nb_actions))
  actor.add(Activation('linear'))
  return actor

def get_lstm_actor(env, nb_actions, window_length, activation='tanh'):
  actor = Sequential()
  actor.add(TimeDistributed(Flatten(input_shape=(1,window_length,) + env.observation_space.shape)))
  actor.add(TimeDistributed(
    Dense(FF_LAYER_SIZE,
      W_regularizer=l2(OPTION_W_REG),
      activity_regularizer=activity_l2(OPTION_A_REG))))
  actor.add(TimeDistributed(Activation('relu')))
  actor.add(LSTM(LSTM_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  #actor.add(Activation('relu'))
  #actor.add(Dense(FF_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG), activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(nb_actions))
  actor.add(Activation(activation))
  return actor

def get_lstm_validation_actor(env, nb_actions, window_length, activation='tanh'):
  actor = Sequential()
  actor.add(TimeDistributed(
    Dense(FF_LAYER_SIZE,
      W_regularizer=l2(OPTION_W_REG),
      activity_regularizer=activity_l2(OPTION_A_REG)),
      input_shape=(window_length,) + env.observation_space.shape,
    ))
  actor.add(TimeDistributed(Activation('relu')))
  actor.add(LSTM(LSTM_LAYER_SIZE, W_regularizer=l2(OPTION_W_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(nb_actions))
  actor.add(Activation(activation))
  return actor

def ddpg_get_actor(env, nb_actions):
  actor = Sequential()
  actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
  actor.add(Dense(FF_LAYER_SIZE,
    W_regularizer=l2(OPTION_W_REG_L1),
    activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(FF_LAYER_SIZE,
    W_regularizer=l2(OPTION_W_REG),
    activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(FF_LAYER_SIZE,
    W_regularizer=l2(OPTION_W_REG),
    activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(nb_actions))
  actor.add(Activation('tanh'))
  return actor

def ddpg_get_validation_actor(env, nb_actions):
  actor = Sequential()
  actor.add(Dense(FF_LAYER_SIZE,
    input_shape=env.observation_space.shape,
    W_regularizer=l2(OPTION_W_REG_L1),
    activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(FF_LAYER_SIZE,
    W_regularizer=l2(OPTION_W_REG),
    activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(FF_LAYER_SIZE,
    W_regularizer=l2(OPTION_W_REG),
    activity_regularizer=activity_l2(OPTION_A_REG)))
  actor.add(Activation('relu'))
  actor.add(Dense(nb_actions))
  actor.add(Activation('tanh'))
  return actor

def ddpg_get_critic(env, nb_actions):
  action_input = Input(shape=(nb_actions,), name='action_input')
  observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
  flattened_observation = Flatten()(observation_input)
  x = merge([action_input, flattened_observation], mode='concat')
  x = Dense(2*FF_LAYER_SIZE)(x)
  x = Activation('relu')(x)
  x = Dense(2*FF_LAYER_SIZE)(x)
  x = Activation('relu')(x)
  x = Dense(2*FF_LAYER_SIZE)(x)
  x = Activation('relu')(x)
  x = Dense(1)(x)
  x = Activation('linear')(x)
  critic = Model(input=[action_input, observation_input], output=x)
  return critic, action_input

def ddpg_get_validation_critic(env, nb_actions):
  action_input = Input(shape=(nb_actions,), name='action_input')
  observation_input = Input(shape=env.observation_space.shape, name='observation_input')
  x = merge([action_input, observation_input], mode='concat')
  x = Dense(2*FF_LAYER_SIZE)(x)
  x = Activation('relu')(x)
  x = Dense(2*FF_LAYER_SIZE)(x)
  x = Activation('relu')(x)
  x = Dense(2*FF_LAYER_SIZE)(x)
  x = Activation('relu')(x)
  x = Dense(1)(x)
  x = Activation('linear')(x)
  critic = Model(input=[action_input, observation_input], output=x)
  return critic, action_input


