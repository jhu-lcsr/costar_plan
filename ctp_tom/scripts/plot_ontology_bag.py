#!/usr/bin/env python

import rosbag
import rospy
from matplotlib import pyplot as plt

def _parse_args():
    return {'filename': "learning_2017-12-06-22-05-30.bag"}

def _main(args):
    objs = {}
    topic = "/br/learning/"
    #for topic, msg, 


if __name__ == "__main__":
    args = _parse_args()
    _main(args)
