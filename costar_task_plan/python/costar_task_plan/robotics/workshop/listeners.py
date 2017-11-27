from __future__ import print_function

import numpy as np
import rospy

from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo


class ListenerManager(object):
    '''
    Listens for data on topics:
      - camera/depth/points
      - joint_states

    Also computes the following transforms from TF:
      - world --> end effector tip (when joint states are received)
      - world --> camera (when point cloud is received)
    '''
    
    def __init__(self,
          camera_topic="camera/depth_registered/points",
          camera_depth_info_topic="camera/depth/info",
          camera_rgb_info_topic="camera/rgb/info",
          joints_topic="joint_states",
          messages_topic="/costar/messages",
          world_frame="/base_link",
          camera_frame="camera_depth_frame",
          ee_frame="/endpoint"):
        '''
        Define the listener manager. By default htese are listed as relative topics
        so they can be easily reconfigured via ROS command-line remapping.

        Parameters:
        ----------
        camera_topic: topic on which RGB-D data is published
        joints_topic: topic on which we receive joints information from the robot
        messages_topic: 
        world_frame: TF frame which we should save all other parameters
                     relative to.
        camera_frame: name of the TF frame for the camera
        ee_frame: name of the end of the kinematic chain for the robot -- should
                  hopefully be the grasp frame, but can also be the wrist joint.
        '''
        self.T_world_ee = None
        self.T_world_camera = None
        self.camera_frame = camera_frame
        self.ee_frame = ee_frame

        self.q = None
        self.dq = None
        self.pc = None
        self.camera_depth_info = None
        self.camera_rgb_info = None

        self._camera_sub = rospy.Subscriber(camera_topic, PointCloud2, self._cloudCb)
        self._camera_depth_info_sub = rospy.Subscriber(camera_depth_info_topic, CameraInfo, self._depthInfoCb)
        self._camera_rgb_info_sub = rospy.Subscriber(camera_rgb_info_topic, CameraInfo, self._rgbInfoCb)
        self._camera_info_sub = rospy.Subscriber(camera_topic, PointCloud2, self._cloudCb)
        self._joints_sub = rospy.Subscriber(joints_topic, JointState, self._jointsCb)
        self._messages_sub = rospy.Subscriber(joints_topic, JointState, self._jointsCb)
        
        self._resetData()

        self.num_trials = 0

    def _resetData(self):
        self.data = {}
        self.data["q"] = []
        self.data["dq"] = []
        self.data["T_ee"] = []
        self.data["T_camera"] = []

        # numpy matrix of xyzrgb values
        self.data["xyzrgb"] = []

        # -------------------------
        # Camera info fields

    def _jointCb(self, msg):
        self.q = msg.position
        self.dq = msg.velocity

    def _cameraCb(self, msg):
        '''
        Parse points and fields from a pointcloud2 message. For more info:
          https://answers.ros.org/question/208834/read-colours-from-a-pointcloud2-python/

        Parameters:
        -----------
        msg: ros sensor_msgs/PointCloud2 message
        '''
        gen = PointCloud2.read_points(msg, skip_nans=True)
        pass

    def _depthInfoCb(self, msg):
        self.camera_depth_info = msg

    def _rgbInfoCb(self, msg):
        self.camera_rgb_info = msg

    def _messageCb(self, msg):
        '''
        This callback parses messages from the main server and writes them.
        '''
        pass

    def update(self):
        '''
        Whenever you call the update function, data is written from currently
        saved messages and other information into the dictionary of saved data.
        '''
        pass

    def write(self):
        '''
        Write to file
        '''
        self.num_trials += 1
        filename = self.filename_template%self.num_trials
        np.savez(filename, **self.data)
        self._resetData()

