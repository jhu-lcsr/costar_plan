from abstract import *
from util import *

import numpy as np

'''
default trainer class
'''
class CemTrainer(AbstractTrainer):

    def __init__(self, env,
        initial_trainer=None,
        initial_model=None,
        noise=1e-1, # noise in initial covariance
        sigma=1e-8, #add to covariance
        elite=None, # number of elite members to choose
        *args, **kwargs):

        # We sum along the whole trajectory to compute rewards.
        super(CemTrainer, self).__init__(env, discount=1.0, *args, **kwargs)

        self.initial_trainer = initial_trainer
        self.initial_model = initial_model

        self.Z = []
        self.noise = noise
        self.actor = None
        self.elite = elite
        self.sigma_noise = sigma

    def _step(self, data, *args, **kwargs):

      # find the best elite set of models
      sum_r = 0
      R = []
      for rollout, _ in data:
        # actually -- path integral RL
        sum_r += np.exp(rollout[0].R)
        R.append(rollout[0].R)

      print "--------------------"
      print "mean = ", np.mean(R)
      print "std = ", np.std(R)
      print "max = ", np.max(R)
      print "min = ", np.min(R)

      for (mu, sigma, _) in self.Z:
        mu *= self.learning_rate
        sigma *= self.learning_rate

      R = [np.exp(r) / sum_r for r in R]
      R = [r  if r > 1e-20 else 0 for r in R]
      print "CEM weights = ", R
      for i, (_, model) in enumerate(data):
        wts = model.get_weights()
        model_wt = R[i]
        for (mu, sigma, shape), wt in zip(self.Z, wts):
          mu += ((1 - self.learning_rate) * model_wt * wt).flatten()
        for (mu, sigma, shape), wt in zip(self.Z, wts):
          dwt = wt.flatten() - mu
          sigma += ((1 - self.learning_rate) * model_wt * np.dot(dwt.T, dwt))
          sigma += np.eye(sigma.shape[0]) * self.sigma_noise

    '''
    Create a new neural net model via sampling from the distribution
    '''
    def sample_model(self):
      weights = []
      for (mu, sigma, shape) in self.Z:
        try:
          wt = mu + np.dot(np.linalg.cholesky(sigma).T,
              np.random.normal(0,1,size=mu.shape))
        except np.linalg.LinAlgError, e:
          print sigma
          print sigma.shape
          raise e
        weights.append(wt.reshape(shape))
      new_actor = clone_model(self.actor)
      new_actor.set_weights(weights)
      return new_actor


    '''
    Sample a particular command from a model
    '''
    def sample_from_model(self, state, model):
      batch = np.array([state])
      raw_action = model.predict(batch)[0]
      return raw_action

    '''
    No compilation needed here --
    '''
    def compile(self, optimizer=None, *args, **kwargs):

      if self.initial_model:
        self.actor = self.initial_model
      elif self.initial_trainer:
        self.initial_trainer.compile(optimizer, *args, **kwargs)
        self.initial_trainer.train(shuffle=True, *args, **kwargs)
        self.actor = self.initial_trainer.getActorModel()
      else:
        raise RuntimeError('Must provide something to initialize CEM')

      self.Z = []
      weights = self.actor.get_weights()
      for w in weights:
        shape = w.shape
        mu = w.flatten()
        sigma = np.eye(mu.shape[0]) * self.noise
        self.Z.append((mu,sigma,shape))

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
        #print "-- ROLLOUT", i,
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
