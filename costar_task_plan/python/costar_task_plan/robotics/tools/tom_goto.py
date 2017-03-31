#!/usr/bin/env python

# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See license for more details

import rospy
from costar_robot import InverseKinematicsUR5
from costar_task_plan.robotics.tom.config import TOM_RIGHT_CONFIG as CONFIG

from sensor_msgs.msg import JointState
import tf
import tf_conversions.posemath as pm

from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from urdf_parser_py.urdf import URDF

def goto(ik, kdl_kin, pub, listener, trans, rot): 

  try:
    tbt, tbr = listener.lookupTransform(
            'torso_link',
            'r_base_link',
            rospy.Time(0))
    bet, ber = listener.lookupTransform(
            'r_base_link',
            'r_ee_link',
            rospy.Time(0))

    T_bt = pm.fromTf((tbt, tbr))
    T_eb = pm.fromTf((bet, ber))
    T = pm.fromTf((trans, rot))

    eet,eer = (0.40920895691643877, -0.46954606404578286, 0.16391727813414897), (0.9200878789467755, -0.36755317350262856, 0.09490153683812817, 0.09662638340086906)

    q0 = [-1.0719114121799995, -1.1008140645600006, 1.7366724169200003,
            -0.8972388608399999, 1.25538042294, -0.028902652380000227,]
    T_fwd = pm.toTf(pm.fromMatrix(kdl_kin.forward(q0)))
    Q = kdl_kin.inverse(T_fwd, q0)
    T_test = pm.fromTf((bet,ber))
    #print pm.toMatrix(T_test) - kdl_kin.forward(q0)
    #Q = ik.solveIK(pm.toMatrix(T_test))
    Q = kdl_kin.inverse(pm.toMatrix(T_test), q0)

    T_pose = pm.toMatrix(T_bt.Inverse() * T)
    print "-----"
    print Q
    print q0
    Q = kdl_kin.inverse(T_pose, q0)
    print Q

    print "to pose", trans, rot
    print "base torso", tbt, tbr
    print "end base", bet, ber
    print "Closest joints =", Q

    msg = JointState(name=CONFIG['joints'],
                       position=Q)
    pub.publish(msg)
  except  (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException), e:
    pass

if __name__ == '__main__':
  rospy.init_node('tom_simple_goto')

  pub = rospy.Publisher('joint_states_cmd', JointState, queue_size=1000)
  ik = InverseKinematicsUR5()

  base_link = 'r_base_link'
  end_link = 'r_ee_link'
  robot = URDF.from_parameter_server()
  tree = kdl_tree_from_urdf_model(robot)
  chain = tree.getChain(base_link, end_link)
  kdl_kin = KDLKinematics(robot, base_link, end_link)

  """
    position: 
      x: 0.648891402264
      y: -0.563835865845
      z: -0.263676911067
    orientation: 
      x: -0.399888401484
      y: 0.916082302699
      z: -0.0071291983402
      w: -0.0288384391252
  """

  rate = rospy.Rate(30)
  listener = tf.TransformListener()
  try:
    while not rospy.is_shutdown():
      goto(ik, kdl_kin, pub, listener, (0.64, -0.56, -0.26), (-0.4, 0.92, -0.01, -0.03))
      rate.sleep()
  except rospy.ROSInterruptException, e:
    pass

