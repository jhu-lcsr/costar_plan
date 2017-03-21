'''
By Chris Paxton
(c) 2017 Johns Hopkins University
See license for details
'''

import copy

from abstract import *
from util import *

import numpy as np

'''
Standard "get weights" function
'''
def nn_get_weights(model):
  return model.get_weights()

def nn_construct(actor, weights):
  new_actor = clone_model(actor)
  new_actor.set_weights(weights)
  return new_actor

def nn_predict(actor, state):
  batch = np.array([state])
  raw_action = actor.predict(batch)[0]
  return raw_action

'''
default trainer class
'''
class CemTrainer(AbstractTrainer):

    def __init__(self, env,
        initial_trainer=None,
        initial_model=None,
        initial_noise=1e-1, # noise in initial covariance
        sigma_regularization=1e-8, #add to covariance
        learning_rate=0.75, # learning rate
        elite=None, # number of elite members to choose
        get_weights_fn=nn_get_weights,
        construct_fn=nn_construct,
        predict_fn=nn_predict,
        *args, **kwargs):

        # We sum along the whole trajectory to compute rewards.
        super(CemTrainer, self).__init__(env,
            discount=1.0,
            learning_rate=learning_rate,
            *args,
            **kwargs)

        self.initial_trainer = initial_trainer
        self.initial_model = initial_model
        self.get_weights_fn = get_weights_fn
        self.construct_fn = construct_fn
        self.predict_fn = predict_fn

        self.Z = []
        self.noise = initial_noise
        self.actor = None
        self.elite = elite
        self.sigma_noise = sigma_regularization

    def _step(self, data, *args, **kwargs):

      # find the best elite set of models
      sum_r = 0
      R = []
      for rollout, _ in data:
        # actually -- path integral RL
        sum_r += np.exp(rollout[0].R)
        R.append(rollout[0].R)

      if self.verbose:
        print "--------------------"
        print "mean = ", np.mean(R)
        print "std = ", np.std(R)
        print "max = ", np.max(R)
        print "min = ", np.min(R)

      R = [np.exp(r) / sum_r for r in R]
      R = [r  if r > 1e-20 else 0 for r in R]
      #print "CEM weights = ", R
      new_Z = []
      for md in self.Z:
        new_Z.append((ModelDistribution(np.zeros(md.mu.shape), np.zeros(md.sigma.shape), md.shape)))

      for i, (_, model) in enumerate(data):
        wts = self.get_weights_fn(model)
        model_wt = R[i]
        #model_wt = (1.0 if R[i] == max(R) else 0.)
        for md, wt in zip(new_Z, wts):
          md.mu += (model_wt * wt).flatten()
        for md, wt in zip(new_Z, wts):
          dwt = np.array([wt.flatten() - md.mu])
          md.sigma += (model_wt * np.dot(dwt.T, dwt))

      #print "UPDATING:",
      for md, new_md in zip(self.Z, new_Z):
        #print md.mu,
        md.mu = (1 - self.learning_rate) * md.mu + \
            self.learning_rate * new_md.mu
        md.sigma = (1 - self.learning_rate) * md.sigma + \
            self.learning_rate * new_md.sigma
        md.sigma += np.eye(md.sigma.shape[0]) * self.sigma_noise
        #print md.mu

    '''
    Create a new neural net model via sampling from the distribution
    '''
    def sample_model(self):
      weights = []
      for md in self.Z:
        try:
          wt = md.mu + np.dot(np.linalg.cholesky(md.sigma).T,
              np.random.normal(0,1,size=md.mu.shape))
        except np.linalg.LinAlgError, e:
          print md.sigma
          print md.sigma.shape
          raise e
        weights.append(wt.reshape(md.shape))

      return self.construct_fn(self.actor, weights)

    '''
    Sample a particular command from a model
    '''
    def sample_from_model(self, state, model):
      return self.predict_fn(model, state)

    '''
    Get the final actor model
    '''
    def getActorModel(self):
      weights = []
      for md in self.Z:
        weights.append(copy.copy(md.mu).reshape(md.shape))
      return self.construct_fn(self.actor, weights)

    '''
    No compilation needed here --
    '''
    def compile(self, *args, **kwargs):

      if self.initial_model is not None:
        self.actor = self.initial_model
      elif self.initial_trainer is not None:
        self.initial_trainer.compile(*args, **kwargs)
        self.initial_trainer.train(shuffle=True)
        self.actor = self.initial_trainer.getActorModel()
      else:
        raise RuntimeError('Must provide something to initialize CEM')

      self.Z = []
      weights = self.get_weights_fn(self.actor)
      for w in weights:
        shape = w.shape
        mu = w.flatten()
        sigma = np.eye(mu.shape[0]) * self.noise
        self.Z.append(ModelDistribution(mu,sigma,shape))

    '''
    Perform however many rollouts need to be executed on each time step.

    This overloads the default _collect_rollouts because we need to sample a
    model for each rollout.
    '''
    def _collect_rollouts(self):
      data = []
      total_reward = 0
      count = 0
      for i in xrange(self.rollouts):
        done = False
        state0 = self.env.reset()
        self._reset_state()
        rollout = []
        length = 0
        model = self.sample_model()
        while not done:
          action0 = self.sample_from_model(state0, model)
          (state1, reward0, done, info) = self.env.step(action0)

          rollout.append(Sample(state0, action0, 1., state1, reward0))
          state0 = state1

          length += 1
          if length >= self.max_rollout_length:
            break

          if self._break:
            break

        if self._break:
          break

        # compute cumulative reward
        self._backup(rollout)
        for ex in rollout:
          total_reward += ex.R
          count += 1
        data.append((rollout, model))
      return data, total_reward, count

    def save(self, filename):
      self.actor.save_weights(filename + "_actor.h5f")

'''
Represents a distribution over parameterized models.
'''
class ModelDistribution(object):
  def __init__(self, mu, sigma, shape):
    self.mu = mu
    self.sigma = sigma
    self.shape = shape
