#!/usr/bin/env python

'''
Read through ontology messages from the virtual reality environment and use 
them to generate a few plots. We want to ensure that these work nicely.
'''

import argparse
import rosbag
import rospy
from matplotlib import pyplot as plt

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename","-f",
                        type=str,
                        default="data.bag")
    return vars(parser.parse_args())

def _main(args):
    obj_history = {}
    left_v = []
    right_v = []
    ontology_msg_topic = "/vr/learning/debugingLearning"
    bag = rosbag.Bag(args['filename'])
    for topic, msg, t in bag:
        if not topic == ontology_msg_topic:
            continue
        for obj in msg.object:
            if not obj in obj_history:
                obj_history[obj.name] = []
            obj_history[obj.name].append(
                    [obj.position.x, obj.position.y, obj.position.z])
        left_v.append([msg.velocity_L.x,
                       msg.velocity_L.y,
                       msg.velocity_L.z])
        right_v.append([msg.velocity_R.x,
                       msg.velocity_R.y,
                       msg.velocity_R.z])
        print msg


if __name__ == "__main__":
    args = _parse_args()
    _main(args)
