import numpy as np

from abstract import *
try:
  import keras.backend as K
except ImportError, e:
  raise ImportError('Actor-critic implementation requires keras!')

'''
default trainer class
'''
class ActorCriticTrainer(AbstractTrainer):

    def __init__(self, env, actor, critic, *args, **kwargs):
        super(ActorCriticTrainer, self).__init__(env, *args, **kwargs)
        self.actor = actor
        self.critic = critic

    '''
    Do a single actor-critic step.
    - accumulate a number of on-policy examples.
    '''
    def _step(self, data):
      pass

    def compile(self, optimizer=None):
      self.actor.compile(optimizer='sgd', loss='mse')
      self.critic.compile(optimizer='sgd', loss='mse')

      # -----------------------------------------------------------------------
      # Lifted from Keras-RL
      # Combine actor and critic so that we can get the policy gradient.
      combined_inputs = [self.critic.input]
      critic_inputs = [self.critic.input]
      #for i in self.critic.input:
      #    if i == self.critic_action_input:
      #        combined_inputs.append(self.actor.output)
      #    else:
      #        combined_inputs.append(i)
      #        critic_inputs.append(i)
      combined_output = self.critic(combined_inputs)
      if K._BACKEND == 'tensorflow':
          grads = K.gradients(combined_output, self.critic.trainable_weights)
          grads = [g / float(self.batch_size) for g in grads]  # since TF sums over the batch
      elif K._BACKEND == 'theano':
          import theano.tensor as T
          grads = T.jacobian(combined_output.flatten(), self.actor.trainable_weights)
          grads = [K.mean(g, axis=0) for g in grads]
      else:
          raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))

      raise NotImplementedError('I have not finished implementing actor-critic. You might want to try something else.')

    def _sample(self, state):
      batch = np.array([[state]])
      raw_action = self.actor.predict_on_batch(batch)
      print "-", raw_action,

      # add noise to the action?

      return raw_action.flatten()

    def _reset_state(self):
      self.actor.reset_states()
