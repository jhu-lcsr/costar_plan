#!/usr/bin/env python

"""
(c) 2016 Chris Paxton
"""

from task_tree_search import *

four_lane_map = HorizontalRoadMap(lanes_per_direction=2,oneway=False,sidewalk=True)
print four_lane_map
world = World(worldmap=four_lane_map)
world.addActor(StaticActor(State(5,1,1,0),"1"))
world.addActor(StaticActor(State(6,1,1,0),"1"))
world.addActor(DefaultActor(State(3,2,1,0),"3"))
world.addActor(StaticActor(State(2,3,3,0),"2"))
world.addActor(StaticActor(State(3,3,3,0),"2"))

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
