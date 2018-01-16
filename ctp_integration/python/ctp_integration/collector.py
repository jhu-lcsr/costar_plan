
import PyKDL as kdl
import rospy 
import tf
import tf_conversions.posemath as pm

from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.h5f import H5fDataset

class DataCollector(object):
    '''
    Manages data collection. Will consume:
    - images from camera
    - depth data (optional)
    - current end effector pose
    - current joint states
    - current gripper status
    '''
    self.js_topic = "joint_states"
    self.rgb_topic = "camera/rgb/image_raw"
    self.depth_topic = "camera/depth_registered/image_raw"
    self.ee = "/endpoint"
    self.base_link = "/base_link"
    self.description = "/robot_description"
    self.data_types = ["h5f", "npz"]

    def __init__(self, data_type="h5f", rate=10, data_root="."):
        '''
        Set up the writer (to save trials to disk) and subscribers (to process
        input from ROS and store the current state).
        '''

        self.tf_listener = tf.TransformListener()

        if isinstance(rate, int) or isinstance(rate, float):
            self.rate = rospy.Rate(rate)
        elif isinstance(rate, rospy.Rate):
            self.rate = rate
        else:
            raise RuntimeError("rate data type not supported: %s" % type(rate))

        self.root = data_root
        self.data_type = data_type
        if self.data_type == "h5f":
            self.writer = H5fDataset(root)
        elif self.data_type == "npz":
            self.writer = NpzDataset(root)
        else:
            raise RuntimeError("data type %s not supported" % data_type)

    def save(self, seed, result):
        '''
        Save function that wraps data set access.
        '''

        # for now all examples are considered a success
        self.npz_writer.write(data, seed, result)

if __name__ == '__main__':
    pass


