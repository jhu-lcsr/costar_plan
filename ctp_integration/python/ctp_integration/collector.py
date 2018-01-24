
import PyKDL as kdl
import rospy 
import tf
import tf_conversions.posemath as pm

from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.h5f import H5fDataset

from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

class DataCollector(object):
    '''
    Manages data collection. Will consume:
    - images from camera
    - depth data (optional)
    - current end effector pose
    - current joint states
    - current gripper status
    '''
    js_topic = "joint_states"
    rgb_topic = "camera/rgb/image_raw"
    depth_topic = "camera/depth_registered/image_raw"
    ee = "/endpoint"
    base_link = "/base_link"
    description = "/robot_description"
    data_types = ["h5f", "npz"]

    def __init__(self, robot_config,
            data_type="h5f",
            rate=10,
            data_root=".",
            img_shape=(128,128),
            camera_frame = "/camera_link",):

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
            self.writer = H5fDataset(self.root)
        elif self.data_type == "npz":
            self.writer = NpzDataset(self.root)
        else:
            raise RuntimeError("data type %s not supported" % data_type)

        self.T_world_ee = None
        self.T_world_camera = None
        self.camera_frame = camera_frame
        self.ee_frame = robot_config['end_link']

        self.q = None
        self.dq = None
        self.pc = None
        self.camera_depth_info = None
        self.camera_rgb_info = None
        self.depth_img = None
        self.rgb_img = None

        #self._camera_depth_info_sub = rospy.Subscriber(camera_depth_info_topic, CameraInfo, self._depthInfoCb)
        #self._camera_rgb_info_sub = rospy.Subscriber(camera_rgb_info_topic, CameraInfo, self._rgbInfoCb)
        self._rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._rgbCb)
        self._depth_sub = rospy.Subscriber(self.depth_topic, Image, self._depthCb)
        self._joints_sub = rospy.Subscriber(self.js_topic, JointState, self._jointsCb)
 
        self._resetData()

        self.verbosity = 1

    def _rgbCb(self, msg):
        self.rgb_img = msg

    def _depthCb(self, msg):
        self.depth_img = msg

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

    def _jointsCb(self, msg):
        self.q = msg.position
        self.dq = msg.velocity
        if self.verbosity > 3:
            rospy.loginfo(self.q, self.dq)

    def save(self, seed, result):
        '''
        Save function that wraps data set access.
        '''

        # for now all examples are considered a success
        self.npz_writer.write(self.data, seed, result)
        self._resetData()

if __name__ == '__main__':
    pass


