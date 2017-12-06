#!/usr/bin/env python

import PyKDL as kdl
import rospy

from costar_task_plan.robotics.perception import TransformIntegator

if __name__ == '__main__':

    rospy.init_node('table_integrator')   

    t06 = kdl.Vector(0.036, 0.004, -0.024)
    R06 = kdl.Rotation.Quaternion(0.001, 0.680, 0.015, 0.733)
    t01 = kdl.Vector(0.001, 0.192, 0.001)
    R01 = kdl.Rotation.Quaternion(-0.002, 0.006, -0.012, 1.000)

    integrator = TransformIntegator(
            "tom_table",
            "/camera_rgb_optical_frame")
    integrator.addTransform("/ar_marker_0", kdl.Frame())
    integrator.addTransform("/ar_marker_1", kdl.Frame(R01,t01))
    integrator.addTransform("/ar_marker_6", kdl.Frame(R06,t06))

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        integrator.tick()
        rate.sleep()
