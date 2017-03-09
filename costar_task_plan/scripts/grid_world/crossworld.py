#!/usr/bin/env python

"""
(c) 2016 Chris Paxton
"""

import numpy as np
import task_tree_search as tts
from task_tree_search import *

world = GetCrossworld()
world.addActor(StaticActor(State(5,9,1,0),"1"))
world.addActor(StaticActor(State(6,9,1,0),"1"))
world.addActor(DefaultActor(State(3,10,3,0),"3"))
world.addActor(DefaultActor(State(9,10,3,0),"4"))
world.addActor(DefaultActor(State(34,0,2,0),"5"))
world.addActor(DefaultActor(State(36,15,0,0),"6"))
world.addActor(StaticActor(State(2,11,3,0),"2"))
world.addActor(StaticActor(State(3,11,3,0),"2"))

tg = TerminalGraphics()
tg.drawWorld(world)
try:
    while True:
        tg.wait()
        world.tick()
        tg.drawWorld(world)
except KeyboardInterrupt, e:
    pass
finally:
    tg.close()
