#!/usr/bin/env python

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import rospy

from costar_task_plan.robotics.core import RosTaskParser
from costar_task_plan.robotics.tom import *


def getArgs():
    parser = argparse.ArgumentParser(add_help=True, description="Parse rosbag into graph.")
    parser.add_argument("bagfile",
            help="name of file or comma-separated list of files")
    parser.add_argument("--demo_topic",
                        help="topic on which demonstration info was published",
                        default="/vr/learning/getDemonstrationInfo")
    parser.add_argument("--task_topic",
                        help="topic on which task info was published",)
    parser.add_argument("--fake",
                        action="store_true",
                        help="create some fake options for stuff")
    parser.add_argument("--project",
                        default=None,
                        help="Project directory to save to",)
    parser.add_argument("--show",
                        action="store_true",
                        help="show a graph of the compiled task")
    parser.add_argument("--debug",
                        action="store_true",
                        help="publish debugging messages")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="print out a ton of information")
    return parser.parse_args()

def fakeTaskArgs():
  '''
  Set up a simplified set of arguments. These are for the optimization loop, 
  where we expect to only have one object to grasp at a time.
  '''
  args = {
    'orange': ['orange_1', 'orange_2', 'orange_3'],
    'box': ['box'],
    'drill': ['drill'],
    'drill_receptacle': ['drill_receptacle'],
    'block': ['block_1', 'block_2'],
    #'endpoint': ['l_ee_link', 'r_ee_link'],
    'endpoint': ['r_ee_link'],
    'high_table': ['tom_table'],
  }
  return args



def main():
    args = getArgs()
    rospy.init_node('parse_task_model')

    rtp = RosTaskParser(
            filename=args.bagfile,
            configs=[TOM_RIGHT_CONFIG, TOM_LEFT_CONFIG],
            unknown_apply_before=4,
            min_action_length=1,
            demo_topic=args.demo_topic)
    rtp.process() # run through the data and create models
    task = rtp.makeTask()
    world = TomWorld(lfd=rtp.lfd)
    if args.fake:
        world.addObjects(fakeTaskArgs())
        filled_args = task.compile(fakeTaskArgs())

        if args.verbose:
            print(task.nodeSummary())
            print(task.children['ROOT()'])

        if args.show:
            from costar_task_plan.tools import showTask
            showTask(task)

        if args.debug:
            try:
                rate = rospy.Rate(30)
                while not rospy.is_shutdown():
                    world.updateObservation()
                    world.debugLfD()
                    rate.sleep()
            except rospy.ROSInterruptException as e:
                pass

    if args.project:
        world.saveModels(args.project)

if __name__ == "__main__":
    main()
