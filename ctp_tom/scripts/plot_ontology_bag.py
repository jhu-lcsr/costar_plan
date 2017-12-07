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
    objs = {}
    topic = 
    #for topic, msg, t


if __name__ == "__main__":
    args = _parse_args()
    _main(args)
