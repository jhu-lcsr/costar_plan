#!/usr/bin/env python

import rospy
import tf

#from moveit_commander import PlanningSceneInterface

from sensor_msgs.msg import JointState
from moveit_msgs.msg import PlanningScene
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point

# This is a very crude simulator that determines what TOM can do. We use this
# thing to publish poses where we want the robot to go.
# It implements the following:
#  - get planning scene
#  - joint trajectory cmd
# And it publishes joint states, too! Wow!
class TomSim(object):

  def __init__(self):
    self.seq = 0
    self.tf_pub = tf.TransformBroadcaster()

    # Create the list of joint names for the simple TOM simulator
    self.joint_names = ['l_front_wheel_joint', 'l_rear_wheel_joint',
        'r_front_wheel_joint', 'r_rear_wheel_joint', 'r_shoulder_pan_joint',
        'r_shoulder_lift_joint', 'r_elbow_joint', 'r_wrist_1_joint',
        'r_wrist_2_joint', 'r_wrist_3_joint', 'l_shoulder_pan_joint',
        'l_shoulder_lift_joint', 'l_elbow_joint', 'l_wrist_1_joint',
        'l_wrist_2_joint', 'l_wrist_3_joint', 'r_gripper_left_finger_joint',
        'r_gripper_left_finger_base_joint', 'r_gripper_right_finger_joint',
        'r_gripper_right_finger_base_joint', 'r_gripper_mid_finger_joint',
        'r_gripper_mid_finger_base_joint', 'l_gripper_left_finger_joint',
        'l_gripper_left_finger_base_joint', 'l_gripper_right_finger_joint',
        'l_gripper_right_finger_base_joint', 'l_gripper_mid_finger_joint',
        'l_gripper_mid_finger_base_joint']

    self.default_pose = [0.0, 0.0, 0.0, 0.20294688542190054,
        -1.0719114121799995, -1.1008140645600006, 1.7366724169200003,
        -0.8972388608399999, 1.25538042294, -0.028902652380000227,
        1.2151680370199998, -1.6210618074000003, -2.05585823016,
        -2.5773626100600002, -1.1008140645600006, -0.8256105484199994,
        0.0026895523773320003, -0.0006283185307176531, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0]

    self.qs = {}
    for name, pose in zip(self.joint_names, self.default_pose):
      self.qs[name] = pose

    self.js_pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
    self.ps_pub = rospy.Publisher('planning_scene', PlanningScene, queue_size=1000)
    self.co_pub = rospy.Publisher('collision_object', CollisionObject, queue_size=1000)

    # Get the first planning scene, containing the loaded geometry and meshes
    # and all that.
    rospy.wait_for_service('/get_planning_scene')

    primitive_table = SolidPrimitive(
            type=SolidPrimitive.BOX, dimensions=[0.6,1.4,0.7])
    pose_table = Pose(position=Point(0.85,0.,-0.06))
    primitive_trash = SolidPrimitive(
            type=SolidPrimitive.BOX, dimensions=[0.2,0.2,0.16])
    pose_trash = Pose(position=Point(0.65,-0.55,0.35))

    table = CollisionObject(id="table",
        primitives=[primitive_table, primitive_trash],
        primitive_poses=[pose_table, pose_trash])
    table.operation = CollisionObject.ADD
    table.header.frame_id = "odom_combined"

    # Collision objects
    self.obstacles = [table]

  def start(self):

    rospy.wait_for_service('/get_planning_scene')

    for co in self.obstacles:
        self.co_pub.publish(co)


  def tick(self):
    self.seq += 1
    msg = JointState(name=self.qs.keys(), position=self.qs.values())
    msg.header.seq = self.seq
    msg.header.stamp = rospy.Time.now()
    self.js_pub.publish(msg)

    # Send set of obstacles so that we can avoid different objects that we
    # already know about.
    #ps_msg = PlanningScene()
    #ps_msg.robot_state.joint_state.name = self.qs.keys()
    #ps_msg.robot_state.joint_state.position = self.qs.values()
    #ps_msg.world.collision_objects = self.obstacles
    #self.ps_pub.publish(ps_msg)

    # Send identity transform from world to odometry frame
    self.tf_pub.sendTransform((0,0,0),
            (0,0,0,1),
            rospy.Time.now(), 
            "/odom_combined",
            "/world")

if __name__ == '__main__':

  rospy.init_node('costar_tom_sim')
  
  # set up sim
  sim = TomSim()

  # wait for move group
  # send initial planning scene updates
  sim.start()

  # loop
  rate = rospy.Rate(30)
  try:
    while not rospy.is_shutdown():
      sim.tick()
      rate.sleep()
  except rospy.ROSInterruptException, e:
    pass
