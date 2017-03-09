#!/usr/bin/env python

import numpy as np
import task_tree_search.needle_master as nm
from task_tree_search.needle_master import NeedleTrajectory

import os

import matplotlib.pyplot as plt

files = os.listdir('trials/')

start_idx = 1
end_idx = 11
envs = [0]*(end_idx-start_idx)
ncols = 5
i = 2

env_filename = "trials/environment_%d.txt"%(i)
trials = []
for file in files:
    if file[:5] == 'trial':
        (env_id,t) = nm.ParseDemoName(file)
        if env_id is i:
            trials.append("trials/" + file)

world = nm.NeedleMasterWorld(env_filename, trials)
s0 = world.sampleStart(from_trial=True) # take a start point from one of our trials

print s0
S0 = NeedleTrajectory([s0]) # needle master is a game of trajectory optimization
a0 = nm.NeedleAction(primitives=3,params=[10,-0.1,5,10,0.1,5,15,0,5])
T = nm.NeedleDynamics(world)

S1 = T(S0,a0)

demo1 = world.trials[0]
print "state0 = ", demo1.s[0]
c0 = nm.NeedleControl(demo1.u[0,0],demo1.u[0,1])
print "demonstration control0 = ", c0.v(), c0.dw()

Tc = nm.NeedleControlDynamics(world)

s1 = Tc(s0,c0)
print "predicted state1 = ", np.array([s1.x(), s1.y(), s1.w()])
print "actual state1 = ", demo1.s[1]

world.show(show_plot=False)
S1.show()
plt.show()
