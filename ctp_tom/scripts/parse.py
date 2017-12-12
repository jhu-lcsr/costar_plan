#!/usr/bin/env python

from costar_task_plan.robotics.representation import RosTaskParser

import argparse
import rospy

def args():
    parser = argparse.ArgumentParser(add_help=True, description="Parse rosbag into graph.")
    parser.add_argument("bagfile", help="name of file")
    return parser.parse_args()

def main():
    args = args()
    rospy.init_node('parse_task_model')

    rtp = RosTaskParser(filename=args.bagfile)
