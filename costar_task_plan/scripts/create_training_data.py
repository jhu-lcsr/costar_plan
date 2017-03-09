#!/usr/bin/env python

import task_tree_search as tts
import numpy as np

world = tts.GetCrossworld()
world.addActor(tts.GetCrossworldDefaultActor(world,"0"))
actor = world.actors[0]
nw = world.getLocalWorld(actor)
features = nw.getFeatures(actor,useIntersection=True,flattened=True)

num_actions = len(world.getActions())
num_worlds = 200
num_timesteps = 150
num_examples = num_worlds*num_timesteps
num_features = features.shape[0]

x = np.zeros((num_examples,num_features))
y = np.zeros((num_examples,num_actions))

idx = 0

# generate random worlds for training data
for num in range(num_worlds):

    print "Getting training data for world %d:"%(num)

    # load crossworld
    world = tts.GetCrossworld()

    # randomly place 10 actors in each world
    for i in range(10):
        world.addActor(tts.GetCrossworldDefaultActor(world,str(i)))

    # take 100 timesteps in each world
    for t in range(num_timesteps):

        #print "\t -- timestep %d"%(t)

        actor = world.actors[0]
        nw = world.getLocalWorld(actor)
        features = nw.getFeatures(actor,useIntersection=True,flattened=True)
        
        world.tick()

        action = actor.last_action.idx

        x[idx,:] = features
        y[idx,action] = 1.0
        idx += 1

print " -- %d total actions"%(num_actions)
print " -- %d examples"%(num_examples)
print x
print y

with open("train.npz","w") as outfile:
    np.savez(outfile,x=x,y=y)
