#!/usr/bin/env python

"""
(c) 2016 Chris Paxton
"""

import numpy as np
import copy
import task_tree_search as tts
from task_tree_search import *

world = GetCrossworld()
for i in range(10):
    world.addActor(GetCrossworldDefaultActor(world,str(i)))

tmp = tts.GetCrossworldDefaultActor(world,"A",bothLanes=True,edgesOnly=True)

tg = TerminalGraphics()

player = tts.PlayerActor(tg,tmp.state)
player.setGoals(tmp.goals)
world.addActor(player)

original = copy.deepcopy(player.state)

tg.drawWorld(world)
try:
    while True:
        world.tick()
        tg.drawWorld(world)

        (code,res) = tts.Evaluate(world,player,i=1)
        tg.writeLine(3,res)

except KeyboardInterrupt, e:
    pass
finally:
    tg.close()
