
import matplotlib.pyplot as plt
import numpy as np
import PyKDL as kdl
import rospy
import tf2_ros as tf2
import tf_conversions.posemath as pm

from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.h5f import H5fDataset
from costar_models.datasets.image import GetJpeg
from costar_models.datasets.image import GetPng
from costar_models.datasets.image import JpegToNumpy
from costar_models.datasets.image import ConvertJpegListToNumpy
from costar_models.datasets.depth_image_encoding import FloatArrayToRgbImage

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from robotiq_c_model_control.msg import CModel_robot_input as GripperMsg

import six
import json
import sys
import datetime
from constants import GetHomeJointSpace
from constants import GetHomePose

def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Apply a timestamp to the front of a filename description.

    see: http://stackoverflow.com/a/5215012/99379
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


class DataCollector(object):
    '''
    Buffers data from an example then writes it to disk. 
    
    Data received includes:
    - images from camera
    - depth data (optional)
    - current end effector pose
    - current joint states
    - current gripper status
    '''

    def __init__(
        self, robot_config,
        task,
        data_type="h5f",
        rate=10,
        data_root=".",
        img_shape=(128,128),
        camera_frame="camera_link",
        tf_buffer=None,
        tf_listener=None,
        action_labels_to_always_log=None,
        verbose=0):


        self.js_topic = "joint_states"
        self.rgb_topic = "/camera/rgb/image_rect_color"
        self.depth_topic = "/camera/depth_registered/hw_registered/image_rect_raw"
        self.ee = "endpoint"
        self.base_link = "base_link"
        self.description = "/robot_description"
        self.data_types = ["h5f", "npz"]
        self.info_topic = "/costar/info"
        self.object_topic = "/costar/SmartMove/object"
        self.gripper_topic = "/CModelRobotInput"
        self.camera_depth_info_topic = "/camera/rgb/camera_info"
        self.camera_rgb_info_topic = "/camera/depth/camera_info"
        self.camera_rgb_optical_frame = "camera_rgb_optical_frame"
        self.camera_depth_optical_frame = "camera_depth_optical_frame"
        self.verbose = verbose
        if action_labels_to_always_log is None:
            self.action_labels_to_always_log = ['move_to_home']
        else:
            self.action_labels_to_always_log = action_labels_to_always_log

        '''
        Set up the writer (to save trials to disk) and subscribers (to process
        input from ROS and store the current state).
        '''
        if tf_buffer is None:
            self.tf_buffer = tf2.Buffer()
        else:
            self.tf_buffer = tf_buffer
        if tf_listener is None:
            self.tf_listener = tf2.TransformListener(self.tf_buffer)
        else:
            self.tf_listener = tf_listener

        if isinstance(rate, int) or isinstance(rate, float):
            self.rate = rospy.Rate(rate)
        elif isinstance(rate, rospy.Rate):
            self.rate = rate
        else:
            raise RuntimeError("rate data type not supported: %s" % type(rate))

        self.root = data_root
        self.data_type = data_type
        rospy.logwarn("Dataset root set to " + str(self.root))
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
        self.rgb_time = None

        self.q = None
        self.dq = None
        self.pc = None
        self.camera_depth_info = None
        self.camera_rgb_info = None
        self.depth_img = None
        self.rgb_img = None
        self.gripper_msg = None

        self._bridge = CvBridge()
        self.task = task
        self.reset()

        self._camera_depth_info_sub = rospy.Subscriber(self.camera_depth_info_topic, CameraInfo, self._depthInfoCb)
        self._camera_rgb_info_sub = rospy.Subscriber(self.camera_rgb_info_topic, CameraInfo, self._rgbInfoCb)
        self._rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._rgbCb)
        self._depth_sub = rospy.Subscriber(self.depth_topic, Image, self._depthCb)
        self._joints_sub = rospy.Subscriber(self.js_topic,
                JointState,
                self._jointsCb)
        self._info_sub = rospy.Subscriber(self.info_topic,
                String,
                self._infoCb)
        self._smartmove_object_sub = rospy.Subscriber(self.object_topic,
                String,
                self._objectCb)
        self._gripper_sub = rospy.Subscriber(self.gripper_topic,
                GripperMsg,
                self._gripperCb)

    def _rgbCb(self, msg):
        if msg is None:
            rospy.loginfo("_rgbCb: msg is None !!!!!!!!!")            
        try:
            self.rgb_time = msg.header.stamp
            cv_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")
            self.rgb_img = np.asarray(cv_image)
            #print(self.rgb_img)
        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def _infoCb(self, msg):
        self.info = msg.data

    def _depthInfoCb(self, msg):
        self.depth_info = msg

    def _rgbInfoCb(self, msg):
        self.rgb_info = msg

    def _objectCb(self, msg):
        self.object = msg.data

    def _gripperCb(self, msg):
        self.gripper_msg = msg

    def _depthCb(self, msg):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg)
            self.depth_img = np.asarray(cv_image)
            #print (self.depth_img)
        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def setTask(self, task):
        self.task = task

    def reset(self):
        self.data = {}
        self.data["nsecs"] = []
        self.data["secs"] = []
        self.data["q"] = []
        self.data["dq"] = []
        self.data["pose"] = []
        self.data["camera"] = []
        self.data["image"] = []
        self.data["depth_image"] = []
        self.data["goal_idx"] = []
        self.data["gripper"] = []
        self.data["label"] = []
        self.data["info"] = []
        self.data["depth_info"] = []
        self.data["rgb_info"] = []
        self.data["object"] = []
        self.data["object_pose"] = []
        self.data["labels_to_name"] = list(self.task.labels)
        self.data["rgb_info_D"] = []
        self.data["rgb_info_K"] = []
        self.data["rgb_info_R"] = []
        self.data["rgb_info_P"] = []
        self.data["rgb_info_distortion_model"] = []
        self.data["depth_info_D"] = []
        self.data["depth_info_K"] = []
        self.data["depth_info_R"] = []
        self.data["depth_info_P"] = []
        self.data["depth_distortion_model"] = []
        self.data["all_tf2_frames_as_yaml"] = []
        self.data["all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json"] = []
        self.data["visualization_marker"] = []
        self.data["camera_rgb_optical_frame_pose"] = []
        self.data["camera_depth_optical_frame_pose"] = []
        #self.data["depth"] = []

        self.info = None
        self.object = None
        self.prev_objects = []
        self.action = None
        self.prev_action = None
        self.current_ee_pose = None
        self.last_goal = 0
        self.prev_last_goal = 0
        self.home_xyz_quat = GetHomePose()

    def _jointsCb(self, msg):
        self.q = msg.position
        self.dq = msg.velocity
        if self.verbose > 3:
            rospy.loginfo(self.q, self.dq)

    def save(self, seed, result):
        '''
        Save function that wraps data set access.

        result: options are 'success' 'failure' or 'error.failure'
        '''
        if self.verbose:
            for k, v in self.data.items():
                print(k, np.array(v).shape)
            print(self.data["labels_to_name"])
            print("Labels and goals:")
            print(self.data["label"])
            print(self.data["goal_idx"])

        if isinstance(result, int) or isinstance(result, float):
            result = "success" if result > 0. else "failure"

        filename = timeStamped("example%06d.%s.h5f" % (seed, result))
        rospy.loginfo('Saving dataset example with filename: ' + filename)
        # for now all examples are considered a success
        self.writer.write(self.data, filename, image_types=[("image", "jpeg"), ("depth_image", "png")])
        self.reset()

    def update(self, action_label, is_done):
        '''
        Compute endpoint positions and update data. Should happen at some
        fixed frequency like 10 hz.

        Parameters:
        -----------
        action: name of high level action being executed
        '''

        switched = False
        if not self.action == action_label:
            if not self.action is None:
                switched = True
            self.prev_action = self.action
            self.action = action_label
            self.prev_objects.append(self.object)
            self.object = None
        if switched or is_done:
            self.prev_last_goal = self.last_goal
            self.last_goal = len(self.data["label"])
            len_label = len(self.data["label"])

            # Count one more if this is the last frame -- since our goal could
            # not be the beginning of a new action
            if is_done:
                len_label += 1
                extra = 1
            else:
                extra = 0

            rospy.loginfo("Starting new action: "
                    + str(action_label)
                    + ", prev was from "
                    + str(self.prev_last_goal)
                    + " to " + str(self.last_goal))
            self.data["goal_idx"] += (self.last_goal - self.prev_last_goal + extra) * [self.last_goal]

            len_idx = len(self.data["goal_idx"])
            if not len_idx  == len_label:
                rospy.logerr("lens = " + str(len_idx) + ", " + str(len_label))
                raise RuntimeError("incorrectly set goal idx")

        # action text to check will be the string contents after the colon
        label_to_check = action_label.split(':')[-1]

        should_log_this_timestep = (self.object is not None or 
                                    label_to_check in self.action_labels_to_always_log)
        if not should_log_this_timestep:
            # here we check if a smartmove object is defined to determine
            # if we should be logging at this time.
            if self.verbose:
                rospy.logwarn("passing -- has not yet started executing motion")
            return True

        if self.verbose:
            rospy.loginfo("Logging: " + str(self.action) +
                    ", obj = " + str(self.object) +
                    ", prev = " + str(self.prev_objects))

        backup_t = rospy.Time(0)
        # get the time for this data sample
        if self.rgb_time is not None:
            t = self.rgb_time
        else:
            t = backup_t
            
        self.t = t
        # make sure we keep the right rgb and depth
        img_jpeg = self.rgb_img
        depth_png = self.depth_img
        # decode the data, this will take some time
        img_jpeg = GetJpeg(img_jpeg)
        depth_png = GetPng(FloatArrayToRgbImage(depth_png))
        have_data = False
        # how many times have we tried to get the transforms
        attempts = 0
        max_attempts = 10
        # the number attempts that should
        # use the backup timestamps
        backup_timestamp_attempts = 4
        while not have_data:
            try:
                c_pose = self.tf_buffer.lookup_transform(self.base_link, self.camera_frame, t)
                ee_pose = self.tf_buffer.lookup_transform(self.base_link, self.ee_frame, t)
                if self.object:
                    obj_pose = self.tf_buffer.lookup_transform(self.base_link, self.object, t)
                rgb_optical_pose = self.tf_buffer.lookup_transform(self.base_link, self.camera_rgb_optical_frame, t)
                depth_optical_pose = self.tf_buffer.lookup_transform(self.base_link, self.camera_depth_optical_frame, t)
                all_tf2_frames_as_string = self.tf_buffer.all_frames_as_string()
                # don't load the yaml because it can take up to 0.2 seconds
                all_tf2_frames_as_yaml = self.tf_buffer.all_frames_as_yaml()
                self.tf2_dict = {}
                transform_strings = all_tf2_frames_as_string.split('\n')
                for transform_string in transform_strings:
                    transform_tokens = transform_string.split(' ')
                    if len(transform_tokens) > 1:
                        k = transform_tokens[1]
                        try:
                            k_pose = self.tf_buffer.lookup_transform(self.base_link, k, t)
    
                            k_xyz_qxqyqzqw = [
                                    k_pose.transform.translation.x,
                                    k_pose.transform.translation.y,
                                    k_pose.transform.translation.z,
                                    k_pose.transform.rotation.x,
                                    k_pose.transform.rotation.y,
                                    k_pose.transform.rotation.z,
                                    k_pose.transform.rotation.w,]
                            self.tf2_dict[k] = k_xyz_qxqyqzqw
                        except (tf2.ExtrapolationException, tf2.ConnectivityException) as e:
                            pass
                
                self.tf2_json = json.dumps(self.tf2_dict)

                have_data = True
            except (tf2.LookupException, tf2.ExtrapolationException, tf2.ConnectivityException) as e:
                rospy.logwarn_throttle('Collector transform lookup Failed: %s to %s, %s, %s'
                                       ' at image time %s and local time %s' %
                                       (self.base_link, self.camera_frame, self.ee_frame, 
                                       str(self.object), str(t), str(backup_t)))
                
                have_data = False
                attempts += 1
                rospy.sleep(0.0)
                if attempts > max_attempts - backup_timestamp_attempts:
                    rospy.logwarn_throttle('Collector failed to use the image '
                                  'rosmsg timestep, trying local ros timestamp as backup.')
                    # try the backup timestamp
                    # even though it will be less accurate
                    t = backup_t
                if attempts > max_attempts:
                    # Could not look up one of the transforms -- either could
                    # not look up camera, endpoint, or object.
                    raise e

        c_xyz_quat = pose_to_vec_quat_list(c_pose)
        rgb_optical_xyz_quat = pose_to_vec_quat_list(rgb_optical_pose)
        depth_optical_xyz_quat = pose_to_vec_quat_list(depth_optical_pose)
        ee_xyz_quat = pose_to_vec_quat_list(ee_pose)
        if self.object:
            obj_xyz_quat = pose_to_vec_quat_list(obj_pose)

        self.current_ee_pose = pm.fromTf(pose_to_vec_quat_pair(ee_pose))

        self.data["nsecs"].append(np.copy(self.t.nsecs)) # time
        self.data["secs"].append(np.copy(self.t.secs)) # time
        self.data["q"].append(np.copy(self.q)) # joint position
        self.data["dq"].append(np.copy(self.dq)) # joint velocuity
        self.data["pose"].append(np.copy(ee_xyz_quat)) # end effector pose (6 DOF)
        self.data["camera"].append(np.copy(c_xyz_quat)) # camera pose (6 DOF)

        if self.object:
            self.data["object_pose"].append(np.copy(obj_xyz_quat))
        elif 'move_to_home' in label_to_check:
            self.data["object_pose"].append(self.home_xyz_quat)
            # TODO(ahundt) should object pose be all 0 when ther eis no object?
            # self.data["object_pose"].append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            raise ValueError("Attempted to log unsupported "
                             "object pose data for action_label " + 
                             str(action_label))
        self.data["camera_rgb_optical_frame_pose"].append(rgb_optical_xyz_quat)
        self.data["camera_depth_optical_frame_pose"].append(depth_optical_xyz_quat)
        #plt.figure()
        #plt.imshow(self.rgb_img)
        #plt.show()
        # print("jpg size={}, png size={}".format(sys.getsizeof(img_jpeg), sys.getsizeof(depth_png)))
        self.data["image"].append(img_jpeg) # encoded as JPEG
        self.data["depth_image"].append(depth_png)
        self.data["gripper"].append(self.gripper_msg.gPO / 255.)

        # TODO(cpaxton): verify
        if not self.task.validLabel(action_label):
            raise RuntimeError("action not recognized: " + str(action_label))

        action = self.task.index(action_label)
        self.data["label"].append(action)  # integer code for high-level action
        self.data["info"].append(np.copy(self.info))  # string description of current step
        self.data["rgb_info_D"].append(self.rgb_info.D)
        self.data["rgb_info_K"].append(self.rgb_info.K)
        self.data["rgb_info_R"].append(self.rgb_info.R)
        self.data["rgb_info_P"].append(self.rgb_info.P)
        self.data["rgb_info_distortion_model"].append(self.rgb_info.distortion_model)
        self.data["depth_info_D"].append(self.depth_info.D)
        self.data["depth_info_K"].append(self.depth_info.K)
        self.data["depth_info_R"].append(self.depth_info.R)
        self.data["depth_info_P"].append(self.depth_info.P)
        self.data["depth_distortion_model"].append(self.depth_info.distortion_model)
        if self.object:
            self.data["object"].append(np.copy(self.object))
        else:
            self.data["object"].append('none')
        self.data["all_tf2_frames_as_yaml"].append(all_tf2_frames_as_yaml)
        self.data["all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json"].append(self.tf2_json)

        return True

def pose_to_vec_quat_pair(c_pose):
    c_xyz = [c_pose.transform.translation.x,
                c_pose.transform.translation.y,
                c_pose.transform.translation.z,]
    c_quat = [c_pose.transform.rotation.x,
                c_pose.transform.rotation.y,
                c_pose.transform.rotation.z,
                c_pose.transform.rotation.w,]
    return c_xyz, c_quat

def pose_to_vec_quat_list(c_pose):
    c_xyz, c_quat = pose_to_vec_quat_pair(c_pose)
    return c_xyz + c_quat

if __name__ == '__main__':
    pass


