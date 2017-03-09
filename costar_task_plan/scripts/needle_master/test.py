#!/usr/bin/env python

import task_tree_search.needle_master as nm
import os

import matplotlib.pyplot as plt

files = os.listdir('trials/')

start_idx = 1
end_idx = 11
envs = [0]*(end_idx-start_idx)
ncols = 5

for i in range(start_idx,end_idx):
    filename = "trials/environment_%d.txt"%(i)

    # process as an environment
    env = nm.Environment(filename)
    envs[i-1] = env
    plt.subplot(2,ncols,i)
    env.Draw()

for file in files:
    if file[:5] == 'trial':
        # process as a trial
        (env,t) = nm.ParseDemoName(file)

        # draw
        if env < end_idx and env >= start_idx:
            filename = "trials/" + file
            demo = nm.Demo(env_height=envs[env-1].height,env_width=envs[env-1].width,filename=filename)
            plt.subplot(2,ncols,env)
            demo.Draw()

plt.show()
