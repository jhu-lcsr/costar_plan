import rospy

from sensor_mgs.msg import JointState



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
      camera_topic="camera/depth/points",
      joints_topic="joint_states",
      camera_frame="camera_depth_frame",
      ee_frame="/endpoint"):
    '''
    Define the listener manager. By default htese are listed as relative topics
    so they can be easily reconfigured via ROS command-line remapping.

    Parameters:
    ----------
    camera_topic: topic on which RGB-D data is published
    joints_topic: topic on which we receive joints information from the robot
    camera_frame: name of the TF frame for the camera
    ee_frame: name of the end of the kinematic chain for the robot -- should
              hopefully be the grasp frame, but can also be the wrist joint.
    '''
    self.T_world_ee = None
    self.T_world_camera = None
    self.camera_frame = camera_frame
    self.ee_frame = ee_frame
    self._camera_sub = rospy.Subscriber(camera_topic, PointCloud2, self._cloudCb)
    self._joints_sub = rospy.Subscriber(joints_topic, JointState, self._jointsCb)

  def _jointCb(self, msg):
    pass

  def _cameraCb(self, msg):
    pass

