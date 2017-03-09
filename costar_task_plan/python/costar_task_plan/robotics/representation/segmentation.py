import rospy


'''
Segmentation
Contains tools and utilities for segmenting data and learning models for different sub-tasks.

fx: features
x: state (world state)
u: action/command/robot state
labels: list of segmentation labels
segment_label: which label to choose
'''

def GetSegment(fx,x,u,labels,segment_label):
    nfx = []
    nx = []
    nu = []

    for i in range(len(labels)):
        if labels[i] == segment_label:
            nfx += [fx[i]]
            nx += [x[i]]
            nu += [u[i]]

    return nfx, nx, nu

