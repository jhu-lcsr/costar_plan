#!/usr/bin/env python

"""
(c) 2016 Chris Paxton
"""

import numpy as np
import task_tree_search as tts
from task_tree_search import *

tg = TerminalGraphics()
try:
    while True:
        world = GetCrossworld()
        for i in range(10):
            world.addActor(GetCrossworldDefaultActor(world,str(i)))
        tg.drawWorld(world)
        tg.wait()
except KeyboardInterrupt, e:
    pass
finally:
    tg.close()
