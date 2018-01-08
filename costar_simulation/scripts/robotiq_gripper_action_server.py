#!/usr/bin/env python

import rospy, actionlib
import thread

from control_msgs.msg import GripperCommandAction
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from math import asin

class ParallelGripperActionController:
   
    def __init__(self):
        rospy.init_node('gripper_controller')
        self.msg = None
       
       
        self.r_pub = rospy.Publisher('gripper_controller/command',
                Float64,
                queue_size=10)
        """
        self.t_pub = rospy.Publisher('arm_controller/command',
                JointTrajectory,
                queue_size=10)
        self.js_sub = rospy.Subscriber('joint_states', JointState,
                self._jointStateCb)
        """

        # subscribe to command and then spin
        self.server = actionlib.SimpleActionServer('~gripper_action', GripperCommandAction, execute_cb=self.actionCb, auto_start=False)
        self.server.start()
        rospy.spin()

    def _jointStateCb(self, msg):
        self.msg = msg
        print msg
        traj = JointTrajectory()
        traj.joint_names = msg.name
        pt = JointTrajectoryPoint()
        pt.positions = list(msg.position)
        pt.positions[1] = 1
        traj.points = [pt]
        self.t_pub.publish(traj)

    def actionCb(self, goal):
        """ Take an input command of width to open gripper. """
        rospy.loginfo('Gripper controller action goal recieved:%f' % goal.command.position)
        command = goal.command.position
        
        # publish msgs
        rmsg = Float64(command)
        
        self.r_pub.publish(rmsg)
        rospy.sleep(3.0)
        self.server.set_succeeded()
        rospy.loginfo('Gripper Controller: Done.')

if __name__=='__main__': 
    try:
        ParallelGripperActionController()
    except rospy.ROSInterruptException:
        pass
