#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym
import time

env = gym.make('LunarLanderContinuous-v2')

for i in xrange(10):
    u = (np.random.rand(2) * 2.0) - 1.0
    (obs, reward, done, info) = env.step(u)

    print "==========================="
    print obs
    print reward
    print done
    print info

    env.render()
    time.sleep(1.0)

env.render(close=True)

