#!/usr/bin/env python

from __future__ import print_function

import rospy
import tf
import PyKDL as kdl
import tf_conversions.posemath as pm

from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import PlanningScene
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse
from trajectory_msgs.msg import JointTrajectoryPoint

class FakeScenePublisher(object):

    def __init__(self):
        self.seq = 0
        self.tf_pub = tf.TransformBroadcaster()
        self.orange1 = (0.641782207489,
                  -0.224464386702,
                  -0.363829042912)
        self.orange2 = (0.69,
                  -0.31,
                  -0.363829042912)
        self.orange3 = (0.68,
                  -0.10,
                  -0.363829042912)
        table_vec = (0, -0.5, 0.863)
        table_rot = kdl.Rotation.RotZ(-np.pi)
        self.table = pm.toTf(kdl.Frame(table_rot, table_vec))

    def tick(self):
        self.tf_pub.sendTransform(self.orange1,
                (0,0,0,1),
                rospy.Time.now(), 
                "/orange_1",
                "/torso_link")
        self.tf_pub.sendTransform(self.orange2,
                (0,0,0,1),
                rospy.Time.now(), 
                "/orange_2",
                "/torso_link")
        self.tf_pub.sendTransform(self.orange3,
                (0,0,0,1),
                rospy.Time.now(), 
                "/orange_3",
                "/torso_link")

if __name__ == '__main__':

    rospy.init_node('costar_tom_sim')

    # set up sim
    sim = FakeScenePublisher()

    # loop
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
          sim.tick()
          rate.sleep()
    except rospy.ROSInterruptException, e:
        pass
