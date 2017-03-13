#!/usr/bin/env python

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
  print "===================="
  #print data
  print "reward =",reward
  print "count =",count

def test(center, guess):
  print "start with: ", np.array(guess)
  print "goal is: ", np.array(center)
  env = PointEnv(np.array(center))
  trainer = CemTrainer(env,
      initial_model=np.array(guess),
      noise=1.e-1,
      rollouts=100,
      learning_rate=0.5,
      steps=100,
      callback=callback,
      get_weights_fn=get_weights,
      construct_fn=construct,
      predict_fn=predict,)
  trainer.compile()
  trainer.train()

if __name__ == '__main__':
  test([0.,0.],[0.4,0.4])
