#!/usr/bin/env python

# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from tom_oranges import MakeTomTaskModel, OrangesTaskArgs

from costar_task_plan.abstract import AbstractReward, AbstractFeatures
from costar_task_plan.mcts import DefaultTaskMctsPolicies, Node
from costar_task_plan.mcts import MonteCarloTreeSearch
from costar_task_plan.mcts import ExecutionPlan, DefaultExecute
from costar_task_plan.robotics.core import *
from costar_task_plan.robotics.tom import TomWorld, OpenLoopTomExecute, ParseTomArgs
from costar_task_plan.tools import showTask
from std_srvs.srv import Empty as EmptySrv

import argparse
import rospy

def load_tom_world(regenerate_models):
    world = TomWorld('./',load_dataset=regenerate_models)
    if regenerate_models:
        world.saveModels('tom')
    return world

class NullReward(AbstractReward):
    def __call__(self, world, *args, **kwargs):
      return 0., 0.
    def evaluate(self, world, *args, **kwargs):
      return 0., 0.
class NullFeatures(AbstractFeatures):
    def __call__(self, world, *args, **kwargs):
      return [0.]

if __name__ == '__main__':
    '''
    Main function:
     - create a TOM world
     - verify that the data is being managed correctly
     - fit models to data
     - create Reward objects as appropriate
     - create Policy objects as appropriate
    '''

    import signal
    signal.signal(signal.SIGINT, exit)

    test_args = ParseTomArgs()

    try:
      rospy.init_node('tom_test_node')
      world = load_tom_world(test_args.regenerate_models)
    except RuntimeError, e:
      print "Failed to create world. Are you in the right directory?"
      raise e

    # Set up the task model
    task = MakeTomTaskModel(world.lfd)
    args = OrangesTaskArgs()
    filled_args = task.compile(args)
    execute = True

    # Perform the search
    objects = ['box1', 'orange1', 'orange2', 'orange3', 'trash1',
            'squeeze_area1']
    debug_objects = {"box":"box1",
                     "orange":"orange1",
                     "trash":"trash1",
                     "squeeze_area":"squeeze_area1"}

    path = do_search(world, task, objects)
    print "Done planning."

