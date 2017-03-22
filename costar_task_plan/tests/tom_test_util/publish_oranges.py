#!/usr/bin/env python

import rospy
import tf

from geometry_msgs.msg import Pose, Point, Quaternion

'''

  x: 0.501931591034
  y: 0.00517476163805
  z: -0.463744103909


  x: 0.496197326183
  y: 0.0104989781976
  z: -0.459818810225


  x: 0.48803137064
  y: 0.00221325131133
  z: -0.445204883814
'''

def get_oranges(i):
    if i == 0:
        oranges = [()]

'''
This script publishes TF frames for a whole set of oranges.
'''
def publish_oranges(oranges):
    pass

def publish_others():
    box = (0.67051013617,
           -0.5828498549,
           -0.280936861547), Pose
    squeeze_area = (0.542672622341,
                    0.013907504104,
                    -0.466499112972)
    trash = (0.29702347941,
             0.0110837137159,
             -0.41238342306)
