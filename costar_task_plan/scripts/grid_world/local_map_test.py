#!/usr/bin/env python

"""
(c) 2016 Chris Paxton
"""

import numpy as np
import task_tree_search as tts
from task_tree_search import *

world = GetCrossworld()
for i in range(10):
    world.addActor(GetCrossworldDefaultActor(world,str(i)))

tg = TerminalGraphics()
tg.drawWorld(world)
try:
    while True:
        tg.wait()
        world.tick()
        tg.drawWorld(world)
        local = world.getLocalWorld(world.actors[0])
        tg.drawWorld(local)
        tg.write(20,0,"action idx = %d"%world.actors[0].last_action.idx)
except KeyboardInterrupt, e:
    pass
finally:
    tg.close()
