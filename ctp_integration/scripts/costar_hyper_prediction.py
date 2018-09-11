#!/usr/bin/env python
""" Run neural network predictions on a real robot, particularly CoSTAR and a block stacking task.

License: Apache v2
"""
import sys
import os
import json
import time
import numpy as np
import tensorflow as tf
import PyKDL as kdl
import rospy
import tf2_ros as tf2
import tf_conversions.posemath as pm
import cv2
from tensorflow.python.platform import flags
import costar_hyper
from costar_hyper import grasp_utilities
from costar_hyper import cornell_grasp_train
from costar_hyper import grasp_model
from costar_hyper import block_stacking_reader
from costar_hyper import grasp_metrics
from threading import Lock
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ctp_integration.ros_geometry import pose_to_vec_quat_list
from ctp_integration.ros_geometry import pose_to_vec_quat_pair
from costar_task_plan.robotics.workshop import UR5_C_MODEL_CONFIG
from skimage.transform import resize
from std_msgs.msg import String
import keras

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


flags.DEFINE_string('load_translation_weights', 'https://github.com/ahundt/costar_dataset/releases/download/v0.1/2018-09-07-22-59-00_train_v0.3_msle_combined_plush_blocks-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3-epoch-226-val_loss-0.000-val_cart_error-0.034.h5.zip',
                    """Path to hdf5 file containing model weights to load and continue training.""")
flags.DEFINE_string('load_translation_hyperparams', 'https://github.com/ahundt/costar_dataset/releases/download/v0.1/2018-09-07-22-59-00_train_v0.3_msle_combined_plush_blocks-nasnet_mobile_semantic_translation_regression_model--dataset_costar_block_stacking-grasp_goal_xyz_3_hyperparams.json',
                    """Load hyperparams from a json file.""")
flags.DEFINE_string('load_rotation_weights', 'https://github.com/ahundt/costar_dataset/releases/download/v0.1/2018-09-07-22-49-31_train_v0.3_msle_combined_plush_blocks-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5-epoch-532-val_loss-0.002-val_angle_error-0.282.h5.zip',
                    """Path to hdf5 file containing model weights to load and continue training.""")
flags.DEFINE_string('load_rotation_hyperparams', 'https://github.com/ahundt/costar_dataset/releases/download/v0.1/2018-09-07-22-49-31_train_v0.3_msle_combined_plush_blocks-vgg_semantic_rotation_regression_model--dataset_costar_block_stacking-grasp_goal_aaxyz_nsc_5_hyperparams.json',
                    """Load hyperparams from a json file.""")

flags.DEFINE_string('translation_problem_type', 'semantic_translation_regression', 'see problem_type parameter in other apis')
flags.DEFINE_string('rotation_problem_type', 'semantic_rotation_regression', 'see problem_type parameter in other apis')
flags.DEFINE_string('force_action', None, 'force predicting only a single action, accepts a string or integer id')
flags.DEFINE_string('default_action', '1', 
    'default action if no action has been'
    ' received from ROS the topic /costar/action_label_current')

FLAGS = flags.FLAGS

def extract_filename_from_url(url):
    # note this is almost certainly insecure, 
    # and the url has to exactly match a filename, 
    # no extra string contents at the end
    filename = url[url.rfind("/")+1:]
    return filename

def get_file_from_url(url, extract=True, file_hash=None, cache_subdir='models'):
    filename = extract_filename_from_url(url)

    found_extension = None
    if extract:
        for extension in ['.tar', '.tar.gz', '.tar.bz', '.zip']:
            if extension in filename:
                found_extension = extension

    path = keras.utils.get_file(filename, url, extract=extract, file_hash=file_hash, cache_subdir=cache_subdir)
    if found_extension is not None: 
        # strip the file extension
        path = path.replace(found_extension, '')

    if not os.path.isfile(path):
        raise ValueError(
            'get_file_from_url() tried extracting the url: ' + str(url) +
            ' and we expected this compression option: ' + str(found_extension) +
            ' and the file directly at the url to match this hash option: ' + str(file_hash) +
            ' . However, the final file is not at the expected location: ' + str(path) +
            ' One possible problem is with compression, it is optional'
            ' but when there is compression we expect'
            ' a filename in the archive that matches the filename in the url.'
            ' You may need to debug the code, or if your use case is different'
            ' try get_file() in Keras.')
    return path

class CostarHyperPosePredictor(object):

    def __init__(
            self,
            label_features_to_extract=None,
            data_features_to_extract=None,
            total_actions_available=41,
            feature_combo_name=None,
            rotation_problem_type=None,
            translation_problem_type=None,
            load_rotation_weights=None,
            load_rotation_hyperparams=None,
            load_translation_weights=None,
            load_translation_hyperparams=None,
            top='classification',
            robot_config=UR5_C_MODEL_CONFIG,
            tf_buffer=None,
            tf_listener=None,
            image_shape=(224, 224, 3),
            force_action=None,
            default_action=None,
            verbose=0):

        if load_rotation_weights is None:
            load_rotation_weights = FLAGS.load_rotation_weights
        if load_rotation_hyperparams is None:
            load_rotation_hyperparams = FLAGS.load_rotation_hyperparams
        if load_translation_weights is None:
            load_translation_weights = FLAGS.load_translation_weights
        if load_translation_hyperparams is None:
            load_translation_hyperparams = FLAGS.load_translation_hyperparams
        if rotation_problem_type is None:
            load_translation_hyperparams = FLAGS.load_translation_hyperparams
        if translation_problem_type is None:
            translation_problem_type = FLAGS.translation_problem_type
        if rotation_problem_type is None:
            rotation_problem_type = FLAGS.rotation_problem_type
        if force_action is None:
            force_action = FLAGS.force_action

        self.using_default_action = True
        if force_action is not None:
            # TODO(ahundt) make this work if the user passes a label string instead of an int
            default_action = force_action
            self.using_default_action = False
        else:
            if default_action is None:
                default_action = FLAGS.default_action

        self.action_labels = [block_stacking_reader.encode_action(default_action, total_actions_available=total_actions_available)]
        self.force_action = force_action

        self.total_actions_available = total_actions_available
        self.label_features_to_extract = label_features_to_extract
        self.data_features_to_extract = data_features_to_extract
        self.need_clear_view_rgb_img = True
        self.clear_view_rgb_img = None

        self.rotation_problem_type = rotation_problem_type
        self.translation_problem_type = translation_problem_type
        self.verbose = verbose
        self.mutex = Lock()
        self.image_shape = image_shape
        self._initialize_ros(robot_config, tf_buffer, tf_listener)

        rotation_hyperparams_path = get_file_from_url(load_rotation_hyperparams)
        rotation_weights_path = get_file_from_url(load_rotation_weights)
        translation_hyperparams_path = get_file_from_url(load_translation_hyperparams)
        translation_weights_path = get_file_from_url(load_translation_weights)

        self.rotation_model, self.rotation_data_features = self._initialize_hypertree_model_for_inference(
                feature_combo_name=feature_combo_name,
                problem_type=rotation_problem_type,
                load_weights=rotation_weights_path,
                load_hyperparams=rotation_hyperparams_path,
                top=top)
        self.translation_model, self.translation_data_features = self._initialize_hypertree_model_for_inference(
                feature_combo_name=feature_combo_name,
                problem_type=translation_problem_type,
                load_weights=translation_weights_path,
                load_hyperparams=translation_hyperparams_path,
                top=top)

    def _initialize_hypertree_model_for_inference(
            self,
            feature_combo_name=None,
            problem_type='semantic_grasp_regression',
            load_weights=None,
            load_hyperparams=None,
            top='classification'):

        if load_weights is None:
            load_weights = FLAGS.load_weights
        if load_hyperparams is None:
            load_hyperparams = FLAGS.load_hyperparams

        if load_weights is None or load_weights == '':
            raise ValueError('A weights file must be specified with: --load_weights path/to/weights.h5f')
        if load_hyperparams is None or load_hyperparams == '':
            raise ValueError('A hyperparams file must be specified with: --load_hyperparams path/to/hyperparams.json')

        # load hyperparams from a file
        hyperparams = grasp_utilities.load_hyperparams_json(
            load_hyperparams, FLAGS.fine_tuning, FLAGS.learning_rate,
            feature_combo_name=feature_combo_name)

        # strip out hyperparams that don't affect the model
        hyperparams.pop('loss', None)
        hyperparams.pop('learning_rate', None)
        hyperparams.pop('checkpoint', None)
        hyperparams.pop('batch_size', None)
        hfcn = hyperparams.pop('feature_combo_name', None)
        if feature_combo_name is None:
            feature_combo_name = hfcn

        [image_shapes, vector_shapes, data_features, model_name,
         monitor_loss_name, label_features, monitor_metric_name,
         loss, metrics, classes, success_only] = cornell_grasp_train.choose_features_and_metrics(feature_combo_name, problem_type)


        model = grasp_model.choose_hypertree_model(
            image_shapes=image_shapes,
            vector_shapes=vector_shapes,
            top=top,
            classes=classes,
            **hyperparams)

        # we don't use the optimizer, so just choose a default
        model.compile(
            optimizer='sgd',
            loss=loss,
            metrics=metrics)
        
        model.summary()

        is_file = os.path.isfile(load_weights)
        if not is_file:
            raise RuntimeError('costar_hyper_prediction.py: Weights file does not exist: ' + load_weights) 
        print(problem_type + ' loading weights: ' + load_weights)
        model.load_weights(load_weights)

        return model, data_features

    def _initialize_ros(self, robot_config, tf_buffer, tf_listener):
        if tf_buffer is None:
            self.tf_buffer = tf2.Buffer(rospy.Duration(120))
        else:
            self.tf_buffer = tf_buffer
        if tf_listener is None:
            self.tf_listener = tf2.TransformListener(self.tf_buffer)
        else:
            self.tf_listener = tf_listener

        # http://wiki.ros.org/depth_image_proc
        # http://www.ros.org/reps/rep-0118.html
        # http://wiki.ros.org/rgbd_launch
        # we will be getting 16 bit integer values in millimeters
        self.rgb_topic = "/camera/rgb/image_rect_color"
        # raw means it is in the format provided by the openi drivers, 16 bit int
        self.depth_topic = "/camera/depth_registered/hw_registered/image_rect"

        self._rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._rgbCb)
        self._bridge = CvBridge()
        self.base_link = "base_link"
        self.rgb_time = None
        self.ee_frame = robot_config['end_link']
        self.labels_topic = '/costar/action_labels'
        self._labels_sub = rospy.Subscriber(
                self.labels_topic,
                String,
                self._labels_Cb)
        self.current_label_topic = '/costar/action_label_current'
        self._current_label_sub = rospy.Subscriber(
                self.current_label_topic,
                String,
                self._current_label_Cb)
        self.info_topic = '/costar/info'
        self._info_sub = rospy.Subscriber(
                self.info_topic,
                String,
                self._info_CB)
        
        # we sleep for 1 second so that
        # the buffer can collect some transforms
        rospy.sleep(1)
        # make sure we can get the transforms we will need to run
        self.get_latest_transform()

    def _info_CB(self, msg):
        """ Update the labels available for actions.

        This also means we have a clear view,
        save the next image as an update to the clear view image.
        """
        if msg is None:
            rospy.logwarn("costar_hyper_prediction()::_info_CB: msg is None !!!!!!!!!")
        else:
            with self.mutex:
                if msg.data == 'STARTING ATTEMPT':
                    self.need_clear_view_rgb_img = True
                    # TODO(ahundt) default starting current label?
                    self.current_label = None

    def _labels_Cb(self, msg):
        """ Update the labels available for actions.

        This also means we have a clear view,
        save the next image as an update to the clear view image.
        """
        if msg is None:
            rospy.logwarn("costar_hyper_prediction()::_rgbCb: msg is None !!!!!!!!!")
        else:
            labels = np.array(json.loads(msg.data))
            if self.verbose:
                print('_labels_Cb() got labels:' + str(labels))
            with self.mutex:
                self.labels = labels

    def _current_label_Cb(self, msg):
        """ Get the list of actions, and encode the current action for the prediction step.
        """
        if msg is None:
            rospy.logwarn("costar_hyper_prediction()::_current_label_Cb: msg is None !!!!!!!!!")
        else:
            current_label = msg.data
            try:
                # TODO(ahundt) incorporate data_features_to_extract, so we use the right encoding method 
                # encode the action
                action_labels = [block_stacking_reader.encode_action(current_label, possible_actions=self.labels)]
            except ValueError as ve:
                rospy.logwarn(
                    "costar_hyper_prediction()::_current_label_Cb: labels list is None,"
                    " can't update the current label yet. Error: " + str(ve))
            with self.mutex:
                self.current_label = current_label
                # only update the action labels if we aren't forcing the action to a particular value
                if self.force_action is None:
                    self.action_labels = action_labels
                    self.using_default_action = False
            if self.verbose:
                print('_current_label_Cb() got:' + str(current_label))

    def _rgbCb(self, msg):
        if msg is None:
            rospy.logwarn("costar_hyper_prediction()::_rgbCb: msg is None !!!!!!!!!")
        try:
            # max out at 10 hz assuming 30hz data source
            if msg.header.seq % 3 == 0:
                cv_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")
                # decode the data, this will take some time

                # rospy.loginfo('rgb color cv_image shape: ' + str(cv_image.shape) + ' depth sequence number: ' + str(msg.header.seq))
                # print('rgb color cv_image shape: ' + str(cv_image.shape))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                # Convert image to numpy format
                rgb_img = np.asarray(cv_image, dtype=np.uint8)
                # Resize to match what is expected of the neural network
                if self.image_shape is not None:
                    rgb_img = resize(rgb_img, self.image_shape, mode='constant', preserve_range=True)

                with self.mutex:
                    self.rgb_time = msg.header.stamp
                    # print('rgb_time stamp: ' + str(msg.header.stamp))
                    self.rgb_img = rgb_img
                    if self.need_clear_view_rgb_img or self.clear_view_rgb_img is None:
                        self.clear_view_rgb_img = rgb_img
                        self.clear_view_rgb_time = msg.header.stamp
                        self.need_clear_view_rgb_img = False

        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def _clearViewCb(self, msg):
        """ We have a clear view, save the next image as an update to the clear view image.
        """
        with self.mutex:
            self.need_clear_view_rgb_img = True
    
    def get_latest_transform(self, from_frame=None, to_frame=None, max_attempts=10, backup_timestamp_attempts=4):
        """
        # Arguments

        max_attempts: The maximum number of times to try getting the transform from ros.
        backup_timestamp_attempts: the number attempts that should use a backup timestamp.
        """
        if from_frame is None:
            from_frame = self.base_link
        if to_frame is None:
            to_frame = self.ee_frame
        local_time = rospy.Time.now()
        # this will get the latest available time
        latest_available_time_lookup = rospy.Time(0)

        ##### BEGIN MUTEX
        with self.mutex:
            # get the time for this data sample
            if self.rgb_time is not None:
                t = self.rgb_time
            else:
                t = local_time

        have_data = False
        # how many times have we tried to get the transforms
        attempts = 0
        # print('all frames: ', self.tf_buffer.all_frames_as_yaml())
        while not have_data:
            try:
                ee_pose = self.tf_buffer.lookup_transform(from_frame, to_frame, t)

                have_data = True
            except (tf2.LookupException, tf2.ExtrapolationException, tf2.ConnectivityException) as e:
                rospy.logwarn_throttle(
                    10.0,
                    'CostarHyperPosePredictor transform lookup Failed: %s to %s,'
                    ' at image time: %s and local time: %s '
                    '\nNote: This message may print >1000x less often than the problem occurs.' %
                    (self.base_link, self.ee_frame, str(t), str(latest_available_time_lookup)))

                have_data = False
                attempts += 1
                # rospy.sleep(0.0)
                if attempts > max_attempts - backup_timestamp_attempts:
                    rospy.logwarn_throttle(
                        10.0,
                        'CostarHyperPosePredictor failed to use the rgb image rosmsg timestamp, '
                        'trying latest available time as backup. '
                        'Note: This message may print >1000x less often than the problem occurs.'
                        ' We checked time t: ' + str(t) + 
                        ', and will now try the latest available: ' + str(latest_available_time_lookup) )
                    # try the backup timestamp even though it will be less accurate
                    t = latest_available_time_lookup
                if attempts > max_attempts:
                    # Could not look up one of the transforms -- either could
                    # not look up camera, endpoint, or object.
                    raise e

        ee_xyz_quat = pose_to_vec_quat_list(ee_pose)
        return ee_xyz_quat

    def __call__(self):
        """ Make the prediction and return the predicted pose
        """
        ee_xyz_quat = self.get_latest_transform()
        with self.mutex:
            rgb_images = [self.rgb_img]
            clear_view_rgb_images = [self.clear_view_rgb_img]
            action_labels = np.copy(self.action_labels)
            if self.using_default_action:
                rospy.logwarn_throttle(
                    10.0,
                    'warning: no user specified action received, '
                    'using default action: ' + str(self.action_labels))

        # encode the prediction information
        X = block_stacking_reader.encode_action_and_images(
                self.translation_data_features,
                poses=[ee_xyz_quat],
                action_labels=action_labels,
                init_images=clear_view_rgb_images,
                current_images=rgb_images)

        translation_predictions = self.translation_model.predict_on_batch(X)

        # encode the prediction information
        X = block_stacking_reader.encode_action_and_images(
                self.rotation_data_features,
                poses=[ee_xyz_quat],
                action_labels=action_labels,
                init_images=clear_view_rgb_images,
                current_images=rgb_images)

        rotation_predictions = self.rotation_model.predict_on_batch(X)

        prediction_xyz_qxyzw = grasp_metrics.decode_xyz_aaxyz_nsc_to_xyz_qxyzw(translation_predictions[0] + rotation_predictions[0])

        # prediction_kdl = kdl.Frame(
        #         kdl.Rotation.Quaternion(prediction_xyz_qxyzw[3], prediction_xyz_qxyzw[4], prediction_xyz_qxyzw[5], prediction_xyz_qxyzw[6]),
        #         kdl.Vector(prediction_xyz_qxyzw[0], prediction_xyz_qxyzw[1], prediction_xyz_qxyzw[2]))
        return prediction_xyz_qxyzw, t


def verify_update_rate(update_time_remaining, update_rate=10, minimum_update_rate_fraction_allowed=0.1, info=''):
    """
    make sure at least 10% of time is remaining when updates are performed.
    we are converting to nanoseconds here since
    that is the unit in which all reasonable
    rates are expected to be measured
    """
    update_duration_sec = 1.0 / update_rate
    minimum_allowed_remaining_time = update_duration_sec * minimum_update_rate_fraction_allowed
    min_remaining_duration = rospy.Duration(minimum_allowed_remaining_time)
    if update_time_remaining < min_remaining_duration:
        rospy.logwarn_throttle(1.0, 'Not maintaining requested update rate, there may be problems with the goal pose predictions!\n'
                               '    Update rate is: ' + str(update_rate) + 'Hz, Duration is ' + str(update_duration_sec) + ' sec\n' +
                               '    Minimum time allowed time remaining is: ' + str(minimum_allowed_remaining_time) + ' sec\n' +
                               '    Actual remaining on this update was: ' + str(float(str(update_time_remaining))/1.0e9) + ' sec\n' +
                               '    ' + info)


def main(_):
    """ Main function, starts predictor.
    """
    print('main')
    rospy.init_node('costar_hyper_prediction')
    predictor = CostarHyperPosePredictor()

    br = tf2.TransformBroadcaster()
    transform_name = 'predicted_goal_' + predictor.ee_frame
    update_rate = 10.0
    rate = rospy.Rate(update_rate)
    progbar = tqdm()

    while not rospy.is_shutdown():
        rate.sleep()
        start_time = time.clock()
        prediction_xyz_qxyzw, prediction_time = predictor()
        tick_time = time.clock()
        vec, quat = pose_to_vec_quat_pair(prediction_xyz_qxyzw)
        br.sendTransform(
            vec,
            quat,
            prediction_time,
            predictor.base_link,
            transform_name)
        update_time = time.clock()

        # figure out where the time has gone
        time_str = ('Total tick + log time: {:04} sec, '
                    'Robot Prediction: {:04} sec, '
                    'Sending Results: {:04} sec'.format(update_time - start_time, tick_time - start_time, update_time - tick_time))
        verify_update_rate(update_time_remaining=rate.remaining(), update_rate=update_rate, info=time_str)
        progbar.update()


if __name__ == '__main__':
    # next FLAGS line might be needed in tf 1.4 but not tf 1.5
    # FLAGS._parse_flags()
    tf.app.run(main=main)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)