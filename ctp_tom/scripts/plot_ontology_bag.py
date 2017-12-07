#!/usr/bin/env python

'''
Read through ontology messages from the virtual reality environment and use 
them to generate a few plots. We want to ensure that these work nicely.
'''

import argparse
import numpy as np
import rosbag
import rospy


from mpl_toolkits.mplot3d import Axes3D
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
            if not obj.name in obj_history:
                obj_history[obj.name] = []
            obj_history[obj.name].append(
                    [obj.position.x, obj.position.y, obj.position.z])
            if obj.name == "orange":
                print t, obj.position.x, obj.position.y, obj.position.z
        left_v.append([msg.velocity_L.x,
                       msg.velocity_L.y,
                       msg.velocity_L.z])
        right_v.append([msg.velocity_R.x,
                       msg.velocity_R.y,
                       msg.velocity_R.z])

    lv = np.array(left_v)
    rv = np.array(right_v)

    print lv.shape
    print rv.shape

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(lv[:,0], lv[:,1], lv[:,2], label='left')
    ax.plot(rv[:,0], rv[:,1], rv[:,2], label='right')
    plt.title('Velocities')
    ax.legend()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for obj, data in obj_history.items():
        if "Controller" in obj:
            continue
        if "Head" in obj:
            continue
        print "----"
        print obj
        print data[0]
        data = np.array(data)
        ax.plot(data[:,0], data[:,1], data[:,2], label=obj)   
    plt.title('Object Positions')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    args = _parse_args()
    _main(args)
