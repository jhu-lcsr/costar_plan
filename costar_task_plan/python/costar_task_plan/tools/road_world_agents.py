try:
  from rl.agents import DDPGAgent, ContinuousDQNAgent
  from rl.agents.dqn import DQNAgent
  from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
  from rl.memory import SequentialMemory
  from rl.random import OrnsteinUhlenbeckProcess
except ImportError, e:
  print "This example uses keras-rl!"
  raise e

import tensorflow as tf

from task_tree_search.gym import SimpleRoadWorldEnv
from task_tree_search.gym import RoadWorldOptionEnv
from task_tree_search.gym import RoadWorldDiscreteSamplerEnv
from task_tree_search.gym import RoadWorldMctsSamplerEnv

from road_world_util import *
from animate import *
from task_tree_search.models.road_world import *

import task_tree_search.mcts as mcts
import task_tree_search.road_world as rw
import task_tree_search.road_world.planning as rwp
import task_tree_search.trainers as tr

def normal_init(shape, name=None):
  return normal(shape, scale=1e-4, name=name)

def setup_cdqn(env, nb_actions):

  memory = SequentialMemory(limit=100000, window_length=1)
  random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.4, size=nb_actions)

  # Build all necessary models: V, mu, and L networks.
  V_model = cdqn_get_v_model(env)
  mu_model = cdqn_get_mu_model(env, nb_actions)
  L_model = cqdn_get_L_model(env, nb_actions)

  agent = ContinuousDQNAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                           memory=memory, nb_steps_warmup=100, random_process=random_process,
                           gamma=.99, target_model_update=1e-3)
  return agent

def setup_ddpg(env, nb_actions, weights_filename=None):

  memory = SequentialMemory(limit=100000, window_length=1)
  random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=0.3, size=nb_actions)

  # Create a simple actor-critic model
  print "Creating simple actor model"
  actor = ddpg_get_actor(env, nb_actions)
  if weights_filename is not None:
    actor.load_weights(weights_filename)

  print "Creating simple critic model"
  critic, action_input = ddpg_get_critic(env, nb_actions)

  agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                    memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                    random_process=random_process, gamma=.99, target_model_update=1e-3,
                    custom_model_objects={"normal_init": normal_init})
  return agent

def get_agent(agent_name,
        env,
        weights_filename=None,
        learning_rate=1e-4,
        clipnorm=1.0,
        load_from_default=True):

  assert len(env.action_space.shape) == 1
  nb_actions = env.action_space.shape[0]
  print "Found an action space of size %d"%nb_actions

  assert(agent_name in agents())
  print "Create Agent"
  if agent_name == "ddpg":
    agent = setup_ddpg(env, nb_actions, weights_filename)
  elif agent_name == "cdqn":
    agent = setup_cdqn(env, nb_actions)
  else:
      raise RuntimeError('Agent %s not recognized!',agent_name)
  agent.compile(Adam(lr=learning_rate, clipnorm=clipnorm), metrics=['mae'])
  return agent
