#!/usr/bin/env python

from __future__ import print_function

import pybullet as pb
pb.connect(pb.GUI)
import matplotlib.pyplot as plt

from run import sim

from costar_task_plan.simulation.parse import ParseBulletArgs

if __name__ == "__main__":
    args = ParseBulletArgs()
    args['task'] = "stack1"
    args['robot'] = "ur5"
    args['features'] = "multi"
    args['model'] = "conditional_image"
    args['verbose'] = True
    args['agent'] = "task"
    if args['profile']:
        import cProfile
        cProfile.run('sim(args)')
    else:
        sim(args)
