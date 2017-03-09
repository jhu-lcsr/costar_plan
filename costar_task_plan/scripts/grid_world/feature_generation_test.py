#!/usr/bin/env python

"""
(c) 2016 Chris Paxton
"""

import numpy as np
import task_tree_search as tts
from task_tree_search import *   
# load crossworld
world = GetCrossworld()

# randomly place 10 actors in each world
for i in range(10):
    world.addActor(GetCrossworldDefaultActor(world,str(i)))

for i in range(10):
    world.tick()

nw = world.getLocalWorld(world.actors[0])

print "Local map (no actors):"
print nw.worldmap

print "Features:"
print nw.getFeatures(world.actors[0],True,True)
