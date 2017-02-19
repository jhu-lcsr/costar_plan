from abstract import *
from util import *
import numpy as np

import task_tree_search.datasets.tools as dt

'''
Default trainer class.

Does a bunch of rollouts, keeps the good ones, and fits a supervised model to
them that will predict actions.
'''
class SupervisedTrainer(AbstractTrainer):

    def __init__(self, policy, actor, R_threshold, *args, **kwargs):
        super(SupervisedTrainer, self).__init__(steps=1, *args, **kwargs)
        assert(self.steps is 1)
        self.policy = policy
        self.actor = actor
        self.R_threshold = R_threshold
        self.train_test_split = 0.8

    def _step(self, data, *args, **kwargs):
      X, Y = [], []
      for samples in data:
        for sample in samples:
          # only include good examples in the training data
          if sample.R > self.R_threshold:
            X.append(sample.f0)
            Y.append(sample.a0.toArray())
      X = np.array(X)
      Y = np.array(Y)

      # get the list of indices
      idx = range(X.shape[0])
      np.random.shuffle(idx)

      # split into train and test data
      num_train = int(self.train_test_split * len(idx))
      train_idx = idx[:num_train]
      test_idx = idx[num_train:]

      X_train = X[train_idx]
      Y_train = Y[train_idx]
      X_test = X[test_idx]
      Y_test = Y[test_idx]

      self.actor.fit(X_train, Y_train,
          validation_data=(X_test, Y_test),
          *args, **kwargs)

    # draw a random action from our current model
    def sample(self, state):
      # actually just ignores the state
      world = self.env._world
      actor = world.getLearner()
      state = actor.state
      action = self.policy.evaluate(world, state, actor)
      return action.toArray(), 1

    def save(self, filename):
      self.actor.save_weights(filename + "_actor.h5f")

    def compile(self,*args,**kwargs):
      self.actor.compile(*args,**kwargs)

    def getActorModel(self):
      return self.actor

    def getAllModels(self):
      return [self.actor]


'''
Modified version for recurrent networks.

We need to modify the step() function primarily
'''
class SupervisedRecurrentTrainer(SupervisedTrainer):

  def __init__(self, policy, actor, R_threshold, window_length,
      *args, **kwargs):
    print kwargs
    super(SupervisedRecurrentTrainer, self).__init__(
        policy,
        actor,
        R_threshold,
        *args, **kwargs)
    if not is_recurrent(self.actor):
      raise RuntimeError('You are using the recurrent trainer with a ' + \
          'non-recurrent model. That is a weird choice.')
    self.window_length = window_length

  def _step(self, data, *args, **kwargs):
    X, Y = None, None
    for samples in data:
      # individual batches
      x, y = [], []
      for sample in samples:
        # only include good examples in the training data
        if sample.R > self.R_threshold:
          x.append(sample.s0)
          y.append(sample.a0)

      if len(x) == 0:
        continue

      x = dt.to_recurrent_samples(x, self.window_length)
      y = np.array(y[self.window_length-1:])
      if X is None:
        X = x
        Y = y
      else:
        X = np.append(X,x,axis=0)
        Y = np.append(Y,y,axis=0)
      
    X = np.array(X)
    Y = np.array(Y)

    # get the list of indices
    idx = range(X.shape[0])
    np.random.shuffle(idx)

    # split into train and test data
    num_train = int(self.train_test_split * len(idx))
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    self.actor.fit(X_train, Y_train,
        validation_data=(X_test, Y_test),
        *args, **kwargs)
