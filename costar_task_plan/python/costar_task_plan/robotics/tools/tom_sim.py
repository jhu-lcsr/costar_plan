#!/usr/bin/env python

import rospy
import tf

from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import PlanningScene
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse
from trajectory_msgs.msg import JointTrajectoryPoint

# This is a very crude simulator that determines what TOM can do. We use this
# thing to publish poses where we want the robot to go.
# It implements the following:
#  - get planning scene
#  - joint trajectory cmd
# And it publishes joint states, too! Wow!
class TomSim(object):

    def reset_cb(self, msg):
        for name, q in zip(self.joint_names, self.default_pose):
            self.qs[name] = q
        return EmptyResponse()

    def __init__(self):
        self.seq = 0
        self.tf_pub = tf.TransformBroadcaster()

        # Create the list of joint names for the simple TOM simulator
        #self.joint_names = ['l_front_wheel_joint', 'l_rear_wheel_joint',
        #    'r_front_wheel_joint', 'r_rear_wheel_joint', 'r_shoulder_pan_joint',
        #    'r_shoulder_lift_joint', 'r_elbow_joint', 'r_wrist_1_joint',
        #    'r_wrist_2_joint', 'r_wrist_3_joint', 'l_shoulder_pan_joint',
        #    'l_shoulder_lift_joint', 'l_elbow_joint', 'l_wrist_1_joint',
        #    'l_wrist_2_joint', 'l_wrist_3_joint', 'r_gripper_left_finger_joint',
        #    'r_gripper_left_finger_base_joint', 'r_gripper_right_finger_joint',
        #    'r_gripper_right_finger_base_joint', 'r_gripper_mid_finger_joint',
        #    'r_gripper_mid_finger_base_joint', 'l_gripper_left_finger_joint',
        #    'l_gripper_left_finger_base_joint', 'l_gripper_right_finger_joint',
        #    'l_gripper_right_finger_base_joint', 'l_gripper_mid_finger_joint',
        #    'l_gripper_mid_finger_base_joint']

        #self.default_pose = [0.0, 0.0, 0.0, 0.20294688542190054,
        #    -1.0719114121799995, -1.1008140645600006, 1.7366724169200003,
        #    -0.8972388608399999, 1.25538042294, -0.028902652380000227,
        #    1.2151680370199998, -1.6210618074000003, -2.05585823016,
        #    -2.5773626100600002, -1.1008140645600006, -0.8256105484199994,
        #    0.0026895523773320003, -0.0006283185307176531, 0.0, 0.0, 0.0, 0.0, 0.0,
        #    0.0, 0.0, 0.0, 0.0, 0.0]

        self.joint_names = ['r_shoulder_lift_joint', 'r_front_wheel_joint',
                'r_gripper_right_finger_joint', 'l_shoulder_pan_joint',
                'l_gripper_left_finger_base_joint', 'l_wrist_2_joint',
                'r_gripper_right_finger_base_joint',
                'l_gripper_mid_finger_base_joint', 'l_gripper_mid_finger_joint',
                'r_wrist_2_joint', 'r_gripper_left_finger_joint',
                'r_gripper_left_finger_base_joint', 'l_wrist_3_joint',
                'l_gripper_right_finger_joint', 'l_elbow_joint',
                'r_gripper_mid_finger_base_joint', 'r_shoulder_pan_joint',
                'r_wrist_3_joint', 'l_shoulder_lift_joint', 'r_rear_wheel_joint',
                'r_elbow_joint', 'l_gripper_right_finger_base_joint',
                'l_gripper_left_finger_joint', 'l_rear_wheel_joint',
                'r_wrist_1_joint', 'l_wrist_1_joint', 'r_gripper_mid_finger_joint',
                'l_front_wheel_joint']
        self.default_pose = [-1.3024941653962576, 0.0, 0.0, 1.2151680370199998,
                0.0, -1.1008140645600006, 0.0, 0.0, 0.0, 2.2992189762402173,
                0.0026895523773320003, -0.0006283185307176531, -0.8256105484199994,
                0.0, -2.05585823016, 0.0, -0.7340859109337838, 1.4271237788102449,
                -1.6210618074000003, 0.20294688542190054, 1.5361204651811313, 0.0,
                0.0, 0.0, -2.0823833025971066, -2.5773626100600002, 0.0, 0.0]

        # These are the preset positions for the various TOM objects. These are 
        # reference frames used for computing features. These are the ones
        # associated with the main TOM dataset.
        self.box = (0.67051013617,
               -0.5828498549,
               -0.280936861547)
        self.squeeze_area = (0.542672622341,
                        0.013907504104,
                        -0.466499112972)
        self.trash = (0.29702347941,
                 0.0110837137159,
                 -0.41238342306)
        self.orange1 = (0.641782207489,
                  -0.224464386702,
                  -0.523829042912)
        self.orange2 = (0.69,
                  -0.31,
                  -0.523829042912)
        self.orange3 = (0.68,
                  -0.10,
                  -0.523829042912)

        # Rotation frame for all of these is pointing down at the table.
        self.rot = (0, 0, 0, 1)
        self.gripper_open = True

        #self.box = pm.toMsg(pm.fromTf((box, rot)))
        #self.trash = pn.toMsg(pm.fromTf((trash, rot)))
        #self.squeeze_area = pm.toMsg(pm.fromTf((trash, rot)))

        self.qs = {}
        for name, pose in zip(self.joint_names, self.default_pose):
          self.qs[name] = pose

        self.reset_srv = rospy.Service('tom_sim/reset', EmptySrv, self.reset_cb)

        self.js_pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
        self.ps_pub = rospy.Publisher('planning_scene', PlanningScene, queue_size=1000)
        self.co_pub = rospy.Publisher('collision_object', CollisionObject, queue_size=1000)
        self.cmd_sub = rospy.Subscriber('joint_states_cmd', JointState, self.cmd_cb)

        # Get the first planning scene, containing the loaded geometry and meshes
        # and all that.
        rospy.wait_for_service('/get_planning_scene')


        primitive_table = SolidPrimitive(
                type=SolidPrimitive.BOX, dimensions=[0.6,1.4,0.7])
        #pose_table = Pose(position=Point(0.85,0.,-0.06))
        pose_table = Pose(position=Point(0.85,0.,-0.06))
        primitive_trash = SolidPrimitive(
                type=SolidPrimitive.BOX, dimensions=[0.2,0.2,0.16])
        #pose_trash = Pose(position=Point(0.65,-0.55,0.35))
        pose_trash = Pose(position=Point(0.65,-0.55,0.35))

        table = CollisionObject(id="table",
            primitives=[primitive_table, primitive_trash],
            primitive_poses=[pose_table, pose_trash])
        table.operation = CollisionObject.ADD
        table.header.frame_id = "odom_combined"

        # Collision objects
        #self.obstacles = [table]
        self.obstacles = []

    def start(self):
        '''
        Publish initial message:
        - wait for a while until move_group is running
        - then publish collision objects so that we can check them and make sure 
          that we don't hit them.
        '''

        rospy.wait_for_service('/get_planning_scene')

        for co in self.obstacles:
            self.co_pub.publish(co)

    def cmd_cb(self, msg):
        '''
        Callback to update the world. Listens to JointState messages so we can get
        the time.
        '''
        if not isinstance(msg, JointState):
            raise RuntimeError('must send a joint state')
        elif len(msg.name) is not len(msg.position):
            raise RuntimeError('number of positions and joint names must match')

        for name, pos in zip(msg.name, msg.position):
            if not name in self.qs:
                raise RuntimeError('world sent joint that does not exist')
            self.qs[name] = pos

    def tick(self):
        '''
        Tick once to update the current world state based on inputs from the world.
        - world needs to send all the right information when it does its own tick.
        Basically, when in execution mode, our dynamics function will send a set
        of joint states here.
        '''

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
        self.tf_pub.sendTransform(self.box,
                (0,0,0,1),
                rospy.Time.now(), 
                "/box1",
                "/torso_link")
        self.tf_pub.sendTransform(self.trash,
                (0,0,0,1),
                rospy.Time.now(), 
                "/trash1",
                "/torso_link")
        self.tf_pub.sendTransform(self.squeeze_area,
                (0,0,0,1),
                rospy.Time.now(), 
                "/squeeze_area1",
                "/torso_link")
        self.tf_pub.sendTransform(self.orange1,
                (0,0,0,1),
                rospy.Time.now(), 
                "/orange1",
                "/torso_link")
        self.tf_pub.sendTransform(self.orange2,
                (0,0,0,1),
                rospy.Time.now(), 
                "/orange2",
                "/torso_link")
        self.tf_pub.sendTransform(self.orange3,
                (0,0,0,1),
                rospy.Time.now(), 
                "/orange3",
                "/torso_link")

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
