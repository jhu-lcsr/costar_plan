import numpy as np

from abstract import *
try:
  import keras.backend as K
  import keras.optimizers as optimizers
except ImportError, e:
  raise ImportError('Reinforce implementation requires keras!')

try:
  import tensorflow as tf
except ImportError, e:
  raise ImportError('Reinforce requires tensorflow!')

'''
Trainer using the REINFORCE rule without a critic/value function estimator.
- actor network directly produces actions
- std network outputs the standard deviation
- critic network outputs a value (optional)

This was designed for use with continuous action spaces, and for Keras stateful
RNNs. 
'''
class ReinforceTrainer(AbstractTrainer):

    def __init__(self,
        env, # environment which we want to analyze
        actor, # network that produces a mean estimate for each time step
        std, # network that produces log probabilities of each observation
        critic, # value function baseline to reduce variable
        reward_scale=1.0, # scale reward if necessary
        reward_baseline=0.0, # static baseline for reward function
        min_std=1e-6,
        *args, **kwargs):
        super(ReinforceTrainer, self).__init__(env, *args, **kwargs)

        self.actor = actor # store RNN model
        self.std = std # store variance model
        self.critic = critic # store value network
        self.grads = [] # store gradients
        self.optimizer = None
        self.reward_scale = reward_scale
        self.reward_baseline = reward_baseline

        self.min_std = min_std
        self.log_min_std = np.log(self.min_std)

    '''
    Do a single actor-critic step.
    - accumulate a number of on-policy examples.
    '''
    def _step(self, data):
      for samples in data:
        #x, r = np.array([sample.s0 for sample in samples]), \
        #       np.array([sample.R for sample in samples])
        #print x.shape, r.shape
        #self.train_fn([x,r,np.zeros(r.shape)])
        #self.train_fn([x, r, 0])
        #print "action =", [s.a0[0] for s in samples]
        #print "state =", [s.s0[0] for s in samples]
        #print "reward =", [s.R for s in samples]
        for sample in samples:
          self.train_fn([[sample.s0], [sample.R], 0])
          

    def compile(self, optimizer=None, *args, **kwargs):

      actor_optimizer = tf.train.AdamOptimizer(self.learning_rate)
      critic_optimizer = tf.train.AdamOptimizer(self.learning_rate)

      self.x = self.actor.input
      self.r = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

      policy_network_params = self.actor.trainable_weights
      value_network_params = self.critic.trainable_weights
      actor_values = self.actor(self.x)
      critic_values = self.critic(self.x)
      std_values = self.std(self.x)

      # Define A3C cost and gradient update equations
      self.a = tf.placeholder("float", [None,] + list(self.env.action_space.shape))
      self.R = tf.placeholder("float", [None, ])

      # A should be a one-hot vector, so this gives us a log probability.
      action_actor_values = tf.reduce_sum(tf.mul(actor_values, self.a), reduction_indices=1)

      # Compile our different networks
      self.actor.compile(optimizer=actor_optimizer, loss='mae')
      self.critic.compile(optimizer=critic_optimizer, loss='mae')

      # Policy network update: our actor is outputting log probabilities, so we
      # don't need the Keras log function here.
      cost_actor = -tf.reduce_sum(action_actor_values) * (self.R - critic_values)
      self.train_actor = actor_optimizer.minimize(cost_actor, var_list=policy_network_params)
      grad_actor = K.gradients(cost_actor, policy_network_params)

      # Value network update
      cost_critic = tf.reduce_mean( tf.square(self.R - critic_values) )
      self.train_critic = critic_optimizer.minimize(cost_critic, var_list=value_network_params)
      grad_critic = K.gradients(cost_critic, value_network_params)

    def sample(self, state):
      batch = np.array([state])
      raw_action = self.actor.predict(batch)
      std = self.std.predict(batch)
      p = 0

      return raw_action.flatten(), p

    def _reset_state(self):
      self.actor.reset_states()

    '''
    Save the actor and critic in different files
    '''
    def save(self, actor_filename, critic_filename, *args, **kwargs):
      pass
