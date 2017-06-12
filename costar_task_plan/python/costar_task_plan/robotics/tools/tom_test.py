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

def load_tom_data_and_run():
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

    # Tom execution works by sending a joint state message with just the robot
    # joints for the arm we want to move. The idea is that we can treat the two
    # arms and the base all as separate "actors."
    plan = ExecutionPlan(path, OpenLoopTomExecute(world, 0))

    reset = rospy.ServiceProxy('tom_sim/reset',EmptySrv)
    rate = rospy.Rate(10)
    try:
        while True:
          # Update observations about the world
          objects = ['box1', 'orange1', 'orange2', 'orange3', 'trash1',
                  'squeeze_area1']
          world.updateObservation(objects)

          # Print out visualization information about the world.
          world.visualize()
          world.visualizePlan(plan)
          world.debugLfD(filled_args[0])

          res = False

          # This world is the observation -- it's not necessarily what the
          # robot is actually going to be changing. Of course, in our case,
          # it totally is.
          if test_args.execute:
            res = plan.step(world)

          if res and test_args.loop:
            reset()
            world.updateObservation(objects)
            path = do_search(world, task, objects)
            plan = ExecutionPlan(path, OpenLoopTomExecute(world, 0))

          rate.sleep()

    except rospy.ROSInterruptException, e:
        pass

def do_search(world, task, objects):

    policies = DefaultTaskMctsPolicies(task)
    search = MonteCarloTreeSearch(policies)

    objects = ['box1', 'orange1', 'orange2', 'orange3', 'trash1', 'squeeze_area1']
    world.updateObservation(objects)
    world = world.fork(world.zeroAction(0))

    for actor in world.actors:
        actor.state.ref = None
        actor.state.seq = 0

    while len(world.observation) == 0:
        rospy.sleep(0.1)
        world.updateObservation(objects)

    print "================================================"
    print "Performing MCTS over options:"
    root = Node(world=world,root=True)
    elapsed, path = search(root,iter=10)
    print "-- ", elapsed, len(path)
    return path

if __name__ == "__main__":
    load_tom_data_and_run()



