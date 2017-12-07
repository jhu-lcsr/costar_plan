#!/usr/bin/env python

'''
Read through ontology messages from the virtual reality environment and use 
them to generate a few plots. We want to ensure that these work nicely.

We want to use the data from the VR and analyze the produced trajectories.
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
    parser.add_argument("--ignore_inputs",
                        help="Do not plot controllers and head.",
                        action="store_true")
    parser.add_argument("--display_object",
                        help="Print out all info about this object.",
                        default=None)
    return vars(parser.parse_args())

def _main(filename, ignore_inputs, display_object, **kwargs):
    obj_history = {}
    left_v = []
    right_v = []
    left_p = []
    right_p = []
    ontology_msg_topic = "/vr/learning/debugingLearning"
    bag = rosbag.Bag(filename)
    for topic, msg, t in bag:
        if not topic == ontology_msg_topic:
            continue
        for obj in msg.object:
            if not obj.name in obj_history:
                obj_history[obj.name] = []
            obj_history[obj.name].append(
                    [obj.position.x, obj.position.y, obj.position.z])
            if obj.name == display_object:
                print t, obj.position.x, obj.position.y, obj.position.z
            if obj.name == "Controller (left)":
                left_p.append([obj.position.x, obj.position.y, obj.position.z])
        left_v.append([msg.velocity_L.x,
                       msg.velocity_L.y,
                       msg.velocity_L.z])
        right_v.append([msg.velocity_R.x,
                       msg.velocity_R.y,
                       msg.velocity_R.z])

    lv = np.array(left_v)
    rv = np.array(right_v)

    lp = np.array(left_p)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(lv[:,0], lv[:,1], lv[:,2], label='left')
    ax.plot(rv[:,0], rv[:,1], rv[:,2], label='right')
    plt.title('Velocities')
    ax.legend()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    print lp
    ax.plot(lp[:,0], lp[:,1], lp[:,2], label= 'left position')
    plt.title('Hand positions')
    ax.legend()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for obj, data in obj_history.items():
        if ignore_inputs:
            if "Controller" in obj:
                continue
            if "Head" in obj:
                continue
        data = np.array(data)
        ax.plot(data[:,0], data[:,1], data[:,2], label=obj)   
    plt.title('Object Positions')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    args = _parse_args()
    _main(**args)
