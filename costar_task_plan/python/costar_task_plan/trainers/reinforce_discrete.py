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
Trainer using the REINFORCE rule.
- actor network directly produces actions
- critic network outputs a value (expected reward)

This was designed for use with discrete action spaces, and possibly for Keras
stateful RNNs. We operate under the assumption that all network outputs are 
linear, so the actor network is outputting log probabilities of its actions.

Based on Andrej Karpathy's code:
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

And on Tim O'Shea's A3C code:
https://github.com/osh/kerlym/blob/master/kerlym/a3c/global_params.py

Both of which were invaluable for figuring out the guts of how to use
Tensorflow to do this.
'''
class DiscreteReinforceTrainer(AbstractTrainer):
  def __init__(self,
      env, # environment to generate training data and to get samples
      actor, # model to determine which discrete action to take
      critic, # value function to reduce variance
      tau=1.,
      clip=(-500., 500.),
      learning_rate = 1e-2,
      *args, **kwargs):

    super(DiscreteReinforceTrainer, self).__init__(env, *args, **kwargs)
    self.actor = actor
    self.critic = critic
    self.tau = tau
    self.clip = clip
    self.train_fn = None
    self.learning_rate = learning_rate

    self.batch_size = 1
    self.nb_actions = self.env.action_space.n

  def _step(self, data):
    sess = K.get_session()
    a = []
    x = []
    r = []
    for samples in data:
      for sample in samples:

        # Convert to one-hot
        a0 = [0]*self.nb_actions
        a0[sample.a0] = 1.

        # append data
        a.append(a0)
        x.append(sample.s0)
        r.append(sample.R)
    
    x = np.array(x)
    a = np.array(a)
    r = np.array(r)

    #uncomment to ensure shapes are what you'd expect
    #print x.shape, a.shape, r.shape

    # update the actor
    self.train_actor.run(
        session=sess,
        feed_dict={
          self.x: x,
          self.R: r,
          self.a: a,
          })

    # update the critic
    self.train_critic.run(
        session=sess,
        feed_dict={
          self.x: x,
          self.R: r,
          self.a: a,
          })

   # draw a random action from our current model
  def sample(self, state):
    batch = np.array([state])
    raw_action = self.actor.predict(batch)[0]
    exp_values = np.exp(np.clip(raw_action / self.tau, self.clip[0], self.clip[1]))
    probs = exp_values / np.sum(exp_values)
    action = np.random.choice(range(len(raw_action)), p=probs)
    return action, raw_action[action]

  def compile(self, *args, **kwargs):
    if not K._BACKEND == 'tensorflow':
      raise RuntimeError('Unsupported Keras backend "{}".'.format(K._BACKEND))

    actor_optimizer = tf.train.AdamOptimizer(self.learning_rate)
    critic_optimizer = tf.train.AdamOptimizer(self.learning_rate)

    self.x = self.actor.input
    self.r = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
    
    policy_network_params = self.actor.trainable_weights
    value_network_params = self.critic.trainable_weights
    actor_values = self.actor(self.x)
    critic_values = self.critic(self.x)

    # Define A3C cost and gradient update equations
    self.a = tf.placeholder("float", [None, self.env.action_space.n])
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

  '''
  Reset trainer state before collecting new rollouts.
  '''
  def _reset_state(self):
    self.actor.reset_states()
    self.critic.reset_states()

  def getActorModel(self):
    return self.actor

  def getCriticModel(self):
    return self.actor

  '''
  Save the actor and critic in different files
  '''
  def save(self, actor_filename, critic_filename, *args, **kwargs):
    pass

