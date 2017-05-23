#!/usr/bin/env python

from costar_task_plan.robotics.core import lfd
from costar_task_plan.robotics.tom import *

if __name__ == "__main__":

    # load world plus DMPs and other things
    world = TomWorld('./',load_dataset=True)

    for name, instances in world.lfd.skill_instances.items():
        features = world.lfd.skill_models[name]


