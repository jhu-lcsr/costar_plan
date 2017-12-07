#!/usr/bin/env python

import numpy as np
import PyKDL as kdl
import rospy

# Load transform and collision object tools
from costar_task_plan.robotics.perception import TransformIntegator
from costar_task_plan.robotics.perception import CollisionObjectManager

if __name__ == '__main__':

    rospy.init_node('table_integrator')   

    t06 = kdl.Vector(0.036, 0.004, -0.024)
    R06 = kdl.Rotation.Quaternion(0.001, 0.680, 0.015, 0.733)
    t01 = kdl.Vector(0.001, 0.192, 0.001)
    R01 = kdl.Rotation.Quaternion(-0.002, 0.006, -0.012, 1.000)

    # Create transform integrator for the table frame.
    integrator = TransformIntegator(
            "tom_table",
            "camera_rgb_optical_frame",
            history_length=50,
            offset=kdl.Frame(
                kdl.Rotation.RotZ(np.pi/2),
                kdl.Vector(0.0225,0,0)))
    integrator.addTransform("/ar_marker_0", kdl.Frame())
    integrator.addTransform("/ar_marker_1", kdl.Frame(R01,t01).Inverse())
    integrator.addTransform("/ar_marker_6", kdl.Frame(R06,t06).Inverse())

    block_offset = kdl.Frame(
            kdl.Rotation.RotZ(-1.*np.pi/2) * kdl.Rotation.RotX(np.pi/2))
    block_offset 
    block_offset *= kdl.Frame(kdl.Rotation(),kdl.Vector(-0.127/4,-0.063/2,-0.063/2+0.045/2+0.01))

    # Create block 1 integrator.
    block_1_integrator = TransformIntegator(
            "block_1",
            "camera_rgb_optical_frame",
            history_length=3,
            listener=integrator.listener,
            broadcaster=integrator.broadcaster,
            offset=kdl.Frame())
    block_1_integrator.addTransform("/ar_marker_5", block_offset)

    # Create block integrator. This just generates a transform for the block 2
    # frame based on observations of the particular marker.
    block_2_integrator = TransformIntegator(
            "block_2",
            "camera_rgb_optical_frame",
            history_length=3,
            listener=integrator.listener,
            broadcaster=integrator.broadcaster,
            offset=kdl.Frame())
    block_2_integrator.addTransform("/ar_marker_4", block_offset)

    # Publish collision objects for all things in the scene.
    manager = CollisionObjectManager(
            root="odom_combined",
            listener=integrator.listener)
    manager.addUrdf("block_1", "/block1_description", "block_1")
    manager.addUrdf("block_2", "/block1_description", "block_2")
    manager.addUrdf("block_3", "/block1_description", "block_3")
    manager.addUrdf("tom_table", "/table_description", "tom_table")

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        integrator.tick()
        block_1_integrator.tick()
        block_2_integrator.tick()
        manager.tick()
        rate.sleep()

