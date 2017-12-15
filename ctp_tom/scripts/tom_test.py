#!/usr/bin/env python

# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from tom_oranges import MakeTomTaskModel, OrangesTaskArgs

from costar_task_plan.abstract import AbstractReward, AbstractFeatures
from costar_task_plan.mcts import DefaultTaskMctsPolicies, Node
from costar_task_plan.mcts import MonteCarloTreeSearch
from costar_task_plan.mcts import PlanExecutionManager, DefaultExecute
from costar_task_plan.robotics.core import *
from costar_task_plan.robotics.tom import TomWorld, OpenLoopTomExecute, ParseTomArgs
from costar_task_plan.tools import showTask
from std_srvs.srv import Empty as EmptySrv

import argparse
import rospy
import sys

test_args = None

def load_tom_world(regenerate_models):
    print "========================================="
    print "Loading world..."
    world = TomWorld('./',load_dataset=regenerate_models)
    if regenerate_models:
        world.saveModels('tom')
    print "done."
    return world

pWorld = None
pTask = None
pObjects = None

def profile_do_search():
    do_search(pWorld, pTask, pObjects)

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

    try:
      rospy.init_node('tom_test_node')
      world = load_tom_world(test_args.regenerate_models)
    except RuntimeError, e:
      print "Failed to create world. Are you in the right directory?"
      raise e


    # Set up the task model
    task = MakeTomTaskModel(world.lfd)
    args = OrangesTaskArgs()
    world.addObjects(args)
    filled_args = task.compile(args)
    execute = True

    # Perform the search
    objects = ['box1', 'orange1', 'orange2', 'orange3', 'trash1',
            'squeeze_area1']
    debug_objects = {"box":"box1",
                     "orange":"orange1",
                     "trash":"trash1",
                     "squeeze_area":"squeeze_area1"}

    if test_args.profile:
        import cProfile
        global pWorld, pTask, pObjects
        pWorld = world
        pTask = task
        pObjects = objects
        cProfile.run("profile_do_search()")
    path = do_search(world, task, objects)

    # Tom execution works by sending a joint state message with just the robot
    # joints for the arm we want to move. The idea is that we can treat the two
    # arms and the base all as separate "actors."

    plan = PlanExecutionManager(path, OpenLoopTomExecute(world, 0))
    reset = rospy.ServiceProxy('tom_sim/reset',EmptySrv)
    rate = rospy.Rate(10)
    try:
        while True:
          # Update observations about the world
          world.updateObservation()

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
            world.updateObservation()
            path = do_search(world, task)
            plan = PlanExecutionManager(path, OpenLoopTomExecute(world, 0))

          rate.sleep()

    except rospy.ROSInterruptException, e:
        pass

def do_search(world, task, objects):
    '''
    Run through a single experiment, generating a trajectory that will satisfy
    all of our conditions and producing a list of policies to execute.
    '''

    policies = DefaultTaskMctsPolicies(task)
    search = MonteCarloTreeSearch(policies)

    world.updateObservation()
    world = world.fork(world.zeroAction(0))

    for actor in world.actors:
        actor.state.ref = None
        actor.state.seq = 0

    while len(world.observation) == 0:
        rospy.sleep(0.1)
        world.updateObservation()

    print "================================================"
    print "Performing MCTS over options:"
    root = Node(world=world,root=True)
    elapsed, path = search(root,iter=10)
    print "-- ", elapsed, len(path)
    return path

if __name__ == "__main__":
    test_args = ParseTomArgs()
    #if test_args.profile:
    #    import cProfile
    #    cProfile.run("load_tom_data_and_run()")
    #else:
    load_tom_data_and_run()



