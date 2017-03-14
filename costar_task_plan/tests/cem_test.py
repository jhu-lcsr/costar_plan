#!/usr/bin/env python

import unittest

import numpy as np

from costar_task_plan.trainers import CemTrainer
from costar_task_plan.gym import PointEnv

def get_weights(model):
    return [model]

def construct(model, weights):
    return weights[0]

def predict(model, state):
    return model

def callback(data, reward, count):
  #print "===================="
  #print data
  print "reward =",reward
  #print "count =",count

def test(center, guess):
  print "start with: ", np.array(guess)
  print "goal is: ", np.array(center)
  env = PointEnv(np.array(center))
  trainer = CemTrainer(env,
      initial_model=np.array(guess),
      initial_noise=1.e+1,
      sigma_regularization=1e-12,
      rollouts=250,
      learning_rate=0.75,
      steps=100,
      callback=callback,
      get_weights_fn=get_weights,
      construct_fn=construct,
      predict_fn=predict,)
  trainer.compile()
  trainer.train()
  return trainer.getActorModel()

class CemTest(unittest.TestCase):

  def test1(self):
    np.random.seed(101)
    res = test([0.,0.,0.],[0.4,0.4,0.4])
    print res
    self.assertLessEqual(np.linalg.norm(res),1e-1)

if __name__ == '__main__':
  unittest.main()
