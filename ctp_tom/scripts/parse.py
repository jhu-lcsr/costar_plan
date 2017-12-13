#!/usr/bin/env python

from costar_task_plan.robotics.core import RosTaskParser
from costar_task_plan.robotics.tom import *

import argparse
import rospy

def getArgs():
    parser = argparse.ArgumentParser(add_help=True, description="Parse rosbag into graph.")
    parser.add_argument("bagfile", help="name of file")
    parser.add_argument("--demo_topic",
                        help="topic on which demonstration info was published",
                        default="/vr/learning/getDemonstrationInfo")
    parser.add_argument("--task_topic",
                        help="topic on which task info was published",)
    parser.add_argument("--fake",
                        action="store_true",
                        help="create some fake options for stuff")
    return parser.parse_args()

def fakeTaskArgs():
  '''
  Set up a simplified set of arguments. These are for the optimization loop, 
  where we expect to only have one object to grasp at a time.
  '''
  args = {
    'orange': ['orange1'],
    'squeeze_area': ['squeeze_area1'],
    'box': ['box1'],
    'trash': ['trash1'],
    'endpoint': ['l_ee_link', 'r_ee_link'],
  }
  return args



def main():
    args = getArgs()
    #rospy.init_node('parse_task_model')

    rtp = RosTaskParser(
            filename=args.bagfile,
            configs=[TOM_RIGHT_CONFIG, TOM_LEFT_CONFIG],
            unknown_apply_before=4,
            min_action_length=3,
            demo_topic=args.demo_topic)
    task = rtp.makeTask()
    world = TomWorld(lfd=rtp.lfd)
    if args.fake:
        filled_args = task.compile(fakeTaskArgs())
        print(task.nodeSummary())
        print(task.children['ROOT()'])

        #from costar_task_plan.tools import showTask
        #showTask(task)

if __name__ == "__main__":
    main()
