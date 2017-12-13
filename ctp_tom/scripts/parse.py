#!/usr/bin/env python

from costar_task_plan.robotics.representation import RosTaskParser
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
    return parser.parse_args()

def main():
    args = getArgs()
    #rospy.init_node('parse_task_model')

    rtp = RosTaskParser(
            filename=args.bagfile,
            configs=[TOM_RIGHT_CONFIG, TOM_LEFT_CONFIG],
            unknown_apply_before=4,
            min_action_length=3,
            demo_topic=args.demo_topic)

if __name__ == "__main__":
    main()
