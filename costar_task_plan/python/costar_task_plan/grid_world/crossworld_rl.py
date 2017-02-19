
'''
(c) 2016 Chris Paxton
'''

from world import *
from actor import *
from crossworld import *

'''
Get a crossworld with 10 random actors placed at different locations
'''


def GetCrossworldForIter(x_var, x_dim, model, useModelActor=True):
    world = GetCrossworld()
    for i in xrange(10):
        world.addActor(
            GetCrossworldDefaultActor(world, str(i), bothLanes=True))

    tmp = GetCrossworldDefaultActor(world, "A", bothLanes=False)
    if useModelActor:
        qact = RegressionActor(tmp.state, sample=False)
        qact.setModel(model, x_var, x_dim)
    else:
        qact = RandomActor(tmp.state)
    qact.setGoals(tmp.goals)
    world.addActor(qact)

    return (world, qact)
