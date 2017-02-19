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
'''
class ReinforceRecurrentTrainer(AbstractTrainer):

    def __init__(self, env, actor, generator, *args, **kwargs):
        super(ReinforceTrainer, self).__init__(env, *args, **kwargs)
        self.actor = actor # store RNN model
        self.generator = generator # store RNN model
        #self.critic = critic 
        self.grads = [] # store gradients
        self.optimizer = None

    '''
    Do a single actor-critic step.
    - accumulate a number of on-policy examples.
    '''
    def _step(self, data):
      pass

    def compile(self, optimizer=None):
      # The loss function doesn't realy matter here, since we're computing our
      # own gradient updates anyway.

      # -----------------------------------------------------------------------
      # Combine actor and critic so that we can get the policy gradient.
      sess = K.get_session()
      self.x = self.actor.input
      self.y = self.actor(self.x)
      self.r = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

      weights = self.actor.trainable_weights # weight tensors
      if K._BACKEND == 'tensorflow':
        gradients = K.gradients(self.y, self.actor.trainable_weights)
        gradients = [g / float(self.batch_size) for g in gradients]  # since TF sums over the batch
      elif K._BACKEND == 'theano':
        import theano.tensor as T
        gradients = T.jacobian(y.flatten(), self.actor.trainable_weights)
        gradients = [K.mean(g, axis=0) for g in gradients]
      else:
        raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))

      # We now have the gradients (`grads`) of the combined model wrt to the actor's weights and
      # the output (`output`). Compute the necessary updates using a clone of the actor's optimizer.
      clipnorm = getattr(optimizer, 'clipnorm', 0.)
      clipvalue = getattr(optimizer, 'clipvalue', 0.)

      # compute real gradients with resepect to reward
      def get_gradients(loss, params):
        assert len(gradients) == len(params)
        modified_grads = [g * self.r for g in gradients]
        if clipnorm > 0.:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in modified_grads]))
            modified_grads = [optimizers.clip_norm(g, clipnorm, norm) for g in modified_grads]
        if clipvalue > 0.:
            modified_grads = [K.clip(g, -clipvalue, clipvalue) for g in modified_grads]
        return modified_grads

      # UNCOMMENT FOR SANITY CHECK
      #tmp = [g * self.r for g in gradients]
      #print sess.run([tmp], {self.x: [[[1,0,1,0,1,0,1,0]]], self.y: [[1,1]], self.r: [0]})

      # set gradient function
      optimizer.get_gradients = get_gradients
      self.optimizer = optimizer

      # SET UP MODEL INPUTS AND OUTPUTS
      # To train, provide x, r, and y.
      # Training op definition:
      self.inputs = [self.x, self.r, K.learning_phase()]
      self.outputs = [self.y]
      updates = self.optimizer.get_updates(self.actor.trainable_weights, self.actor.constraints, None)
      updates += self.actor.updates  # include other updates of the actor, e.g. for BN
      self.train_fn = K.function(self.inputs, self.outputs, updates=updates)

      self.actor.compile(optimizer=optimizer, loss='mse')

      '''
      # UNCOMMENT THIS BLOCK FOR A SANITY CHECK
      # This call is essentially just our forward call, but with the rewards added.
      self.actor.reset_states()
      print self.train_fn([[[[1,0,1,0,1,0,1,0]]],[0],0])
      print self.train_fn([[[[1,0,1,0,1,0,1,0]]],[0],0])
      print self.train_fn([[[[1,0,1,0,1,0,1,0]]],[0],0])
      self.actor.reset_states()
      print self.train_fn([[[[1,0,1,0,1,0,1,0]]],[1],0])
      print self.train_fn([[[[1,0,1,0,1,0,1,0]]],[1],0])
      print self.train_fn([[[[1,0,1,0,1,0,1,0]]],[-1],0])
      self.actor.reset_states()
      print self.train_fn([[[[1,0,1,0,1,0,1,0]]],[1],0])
      print self.train_fn([[[[1,0,1,0,1,0,1,0]]],[1],0])
      print self.train_fn([[[[1,0,1,0,1,0,1,0]]],[-1],0])
      '''

      '''
      OUTPUT:
      For the first three rows, we see that with r=0 there was no change to
      the learned model.
      [array([[-0.04176936,  0.02830091]], dtype=float32)]
      [array([[-0.0702659,  0.0320517]], dtype=float32)]
      [array([[-0.09146999,  0.01462531]], dtype=float32)]
      [array([[-0.04176936,  0.02830091]], dtype=float32)]
      [array([[-0.07756874,  0.02411896]], dtype=float32)]
      [array([[-0.11531718, -0.00939121]], dtype=float32)]
      [array([[-0.05983357,  0.0084069 ]], dtype=float32)]
      [array([[-0.10461193, -0.0051554 ]], dtype=float32)]
      [array([[-0.14364725, -0.03858872]], dtype=float32)]
      '''

    def _sample(self, state):
      batch = np.array([[state]])
      raw_action = self.actor.predict_on_batch(batch)
      print "-", raw_action,

      # add noise to the action?

      return raw_action.flatten()

    def _reset_state(self):
      # Note that for this model we want to do stateful prediction. This means
      # that states are fed from the simulator one at a time. For most examples
      # I would expect these to be the same, but it's completely possible that
      # states could vary between subsequent calls -- for example if there is
      # noise.
      # The reset_states() call below is necessary to reset all our "stateful"
      # RNN layers so that when we query from a new environment it is not
      # contaminated by leftover data from previous trials.
      self.actor.reset_states()
