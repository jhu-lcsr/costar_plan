#!/usr/bin/env python

'''
Read through ontology messages from the virtual reality environment and use 
them to generate a few plots. We want to ensure that these work nicely.

We want to use the data from the VR and analyze the produced trajectories.
'''

import argparse
import numpy as np
from numpy import linalg as LA
import rosbag
import rospy
from scipy import signal

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

#function to compute the distances between the EF and all the objects from VR
def get_distance(obj_history, hand):
    dist = {}
    distance = []
    print len(obj_history), len(hand)
    
    for obj, data in obj_history.items():
        data = np.array(data)
        print data.shape    
        distance.append(np.linalg.norm(hand-data))
                
        print obj
           # dist[obj] = distance
            
    return dist


def _main(filename, ignore_inputs, display_object, **kwargs):
    obj_history = {}
    left_v = []
    right_v = []
    left_p = []
    right_p = []
    left_vn = []
    right_vn =[]
    distance_obj_L = {}
    distance_obj_R = {} 
    ontology_msg_topic = "/vr/learning/debugingLearning"
    bag = rosbag.Bag(filename)
    for topic, msg, t in bag:
        if not topic == ontology_msg_topic:
            continue
        for obj in msg.object:
            if not obj.name in obj_history:
                obj_history[obj.name] = []
                
            if obj.name == "Controller (left)":
                left_p.append([obj.position.x, obj.position.y, obj.position.z])
                
            elif obj.name == "Controller (right)":
                right_p.append([obj.position.x, obj.position.y, obj.position.z])
                
            obj_history[obj.name].append([obj.position.x, obj.position.y, obj.position.z])
            if obj.name == display_object:
                print t, obj.position.x, obj.position.y, obj.position.z
           
        left_v.append([msg.velocity_L.x,
                       msg.velocity_L.y,
                       msg.velocity_L.z])
        Lvel_norm = LA.norm([msg.velocity_L.x, msg.velocity_L.y, msg.velocity_L.z])
        left_vn.append(Lvel_norm)
        right_v.append([msg.velocity_R.x,
                       msg.velocity_R.y,
                       msg.velocity_R.z])
        Rvel_norm = LA.norm([msg.velocity_R.x, msg.velocity_R.y, msg.velocity_R.z])
        right_vn.append(Rvel_norm)

    #remove the controllers (EF) information from the dictionary
    del obj_history["Controller (left)"]
    del obj_history["Controller (right)"]
    del obj_history["Head (eye)"]
  
    lv = np.array(left_v)
    rv = np.array(right_v)

    lp = np.array(left_p)
    rp = np.array(right_p)
    
    print len(obj_history), obj_history.keys()
    
    #compute the distance between the right hand and the objects
    distance_obj_R = get_distance (obj_history, right_p)
    
    #compute the distance between the left hand and the objects
    
    # we need to filter out the velocities, because they are noisy
    b_l, a_l = signal.butter(2, 0.35)
    print b_l, a_l
    
    Lv_filtered = signal.filtfilt(b_l, a_l, left_vn, padlen=50)
    Rv_filtered = signal.filtfilt(b_l, a_l, right_vn, padlen=50)
    
    #Plot the hand velocities normalized
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(left_vn, label = 'left hand')
    ax.plot(Lv_filtered,label = 'left hand filtered')
    ax.plot(right_vn, label = 'right hand')
    ax.plot(Rv_filtered, label = 'right hand filtered')
    ax.set(xlabel='time', ylabel='velocity', title='Normalized velocity')
    ax.grid()
    ax.legend()

     #plot the distances between the hand and all the objects
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for obj, data in distance_obj_R.items():
        #print data
        ax.plot(data, label = obj)
    ax.set(xlabel='time', ylabel='distance', title='Distance between objects and Right hand')
    ax.grid()
    ax.legend()

    #plot the 3d hand velocites
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot(lv[:,0], lv[:,1], lv[:,2], label='left')
    #ax.plot(rv[:,0], rv[:,1], rv[:,2], label='right')
    #plt.title('Velocities')
    #ax.legend()

    #plot the 3D hand positions
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #print lp
    ax.plot(lp[:,0], lp[:,1], lp[:,2], label= 'left position')
    ax.plot(rp[:,0], rp[:,1], rp[:,2], label= 'right position')
    plt.title('Hand positions')
    ax.legend()

    #plot the 3D object positions
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    for obj, data in obj_history.items():
        if "Controller (left)" in obj:
            continue
        if "Controller (right)" in obj:
            continue
        if "Head" in obj:
            continue
        data = np.array(data)
       # print data
        ax.plot(data[:,0], data[:,1], data[:,2], label=obj)   
    plt.title('Object Positions')
    ax.legend()
  
    
    plt.show()

if __name__ == "__main__":
    args = _parse_args()
    _main(**args)
