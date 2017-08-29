"""Code for building the input for the prediction model.

Author: Andrew Hundt <ATHundt@gmail.com>

License: Apache v2 https://www.apache.org/licenses/LICENSE-2.0
"""

import os
import errno
from six import iteritems

import numpy as np
import tensorflow as tf
import re

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.ops import data_flow_ops
from keras.utils import get_file

import moviepy.editor as mpy

tf.flags.DEFINE_string('data_dir',
                       os.path.join(os.path.expanduser("~"),
                                    '.keras', 'datasets', 'grasping'),
                       """Path to dataset in TFRecord format
                       (aka Example protobufs) and feature csv files.""")
tf.flags.DEFINE_string('visualization_dir',
                       os.path.join(os.path.expanduser("~"),
                                    '.keras', 'datasets', 'grasping', 'images_extracted_grasp'),
                       """Path to output data visualizations such as image gifs and ply clouds.""")
tf.flags.DEFINE_integer('batch_size', 2, 'batch size per compute device')
tf.flags.DEFINE_integer('sensor_image_width', 640, 'Camera Image Width')
tf.flags.DEFINE_integer('sensor_image_height', 512, 'Camera Image Height')
tf.flags.DEFINE_integer('sensor_color_channels', 3,
                        'Number of color channels (3, RGB)')
tf.flags.DEFINE_boolean('grasp_download', False,
                        """Download the grasp_dataset to data_dir if it is not already present.""")
tf.flags.DEFINE_string('grasp_dataset', '102',
                       """Filter the subset of 1TB Grasp datasets to run.
                       None by default. 'all' will run all datasets in data_dir.
                       '052' and '057' will download the small starter datasets.
                       '102' will download the main dataset with 102 features,
                       around 110 GB and 38k grasp attempts.
                       See https://sites.google.com/site/brainrobotdata/home
                       for a full listing.""")

FLAGS = flags.FLAGS


def mkdir_p(path):
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class GraspDataset(object):
    """Google Grasping Dataset - about 1TB total size
        https://sites.google.com/site/brainrobotdata/home/grasping-dataset

        Data from https://arxiv.org/abs/1603.02199

        Downloads to `~/.keras/datasets/grasping` by default.

        grasp_listing.txt lists all the files in all grasping datasets;
        *.csv lists the number of features, the number of grasp attempts, and all feature names;
        *.tfrecord and *.tfrecord*-of-* is the actual data stored in the tfrecord.

        TODO(ahundt) update to use new TF Dataset API https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data
        TODO(ahundt) This only supports one dataset at a time (aka 102 feature version or 057)

        # Arguments

        data_dir: Path to dataset in TFRecord format
            (aka Example protobufs) and feature csv files.
             `~/.keras/datasets/grasping` by default.

        dataset: Filter the subset of 1TB Grasp datasets to run.
            None by default, which loads the data specified by the tf flag
            grasp_dataset command line parameter.
            'all' will run all datasets in data_dir.
            '052' and '057' will download the small starter datasets.
            '102' will download the main dataset with 102 features,
            around 110 GB and 38k grasp attempts.
            See https://sites.google.com/site/brainrobotdata/home
            for a full listing.

        download: True to actually download the dataset, also see FLAGS.
    """
    def __init__(self, data_dir=None, dataset=None, download=None):
        if data_dir is None:
            data_dir = FLAGS.data_dir
        self.data_dir = data_dir
        if dataset is None:
            dataset = FLAGS.grasp_dataset
        self.dataset = dataset
        if download is None:
            download = FLAGS.grasp_download
        if download:
            self.download(data_dir, dataset)

    def download(self, data_dir=None, dataset='all'):
        '''Google Grasping Dataset - about 1TB total size
        https://sites.google.com/site/brainrobotdata/home/grasping-dataset

        Downloads to `~/.keras/datasets/grasping` by default.
        Includes grasp_listing.txt with all files in all datasets;
        the feature csv files which specify the dataset size,
        the features (data channels), and the number of grasps;
        and the tfrecord files which actually contain all the data.

        # Arguments

            dataset: The name of the dataset to download, downloads all by default
                with the '' parameter, 102 will download the 102 feature dataset
                found in grasp_listing.txt.

        # Returns

           list of paths to the downloaded files

        '''
        dataset = self._update_dataset_param(dataset)
        if data_dir is None:
            if self.data_dir is None:
                data_dir = FLAGS.data_dir
            else:
                data_dir = self.data_dir
        mkdir_p(data_dir)
        print('Downloading datasets to: ', data_dir)
        listing_url = 'https://sites.google.com/site/brainrobotdata/home/grasping-dataset/grasp_listing.txt'
        grasp_listing_path = get_file('grasp_listing.txt', listing_url, cache_subdir=data_dir)
        grasp_files = np.genfromtxt(grasp_listing_path, dtype=str)
        url_prefix = 'https://storage.googleapis.com/brain-robotics-data/'
        files = [get_file(fpath.split('/')[-1], url_prefix + fpath, cache_subdir=data_dir)
                 for fpath in grasp_files
                 if '_' + dataset in fpath]
        return files

    def _update_dataset_param(self, dataset):
        """Internal function to configure which subset of the datasets is being used.
        Helps to choose a reasonable default action based on previous user parameters.
        """
        if dataset is None and self.dataset is None:
            return []
        if dataset is 'all':
            dataset = ''
        if dataset is None and self.dataset is not None:
            dataset = self.dataset
        return dataset

    def _get_feature_csv_file_paths(self, dataset=None):
        """List feature csv files with full paths in the data_dir.
        Feature csv files identify each dataset, the size, and its data channels.
        One example is: 'features_102.csv'
        """
        dataset = self._update_dataset_param(dataset)
        # print("csv_search: ", os.path.join(os.path.expanduser(self.data_dir), '*{}*.csv'.format(dataset)))
        return gfile.Glob(os.path.join(os.path.expanduser(self.data_dir), '*{}*.csv'.format(dataset)))

    def get_features(self, dataset=None):
        """Get all features associated with the dataset specified when this class
        was constructed. Use when only one dataset will be processed.

        dataset: Filter the subset of 1TB Grasp datasets to run.
            None is the default, which utilizes the currently loaded dataset,
            'all' will run all datasets in data_dir.
            '052' and '057' will download the small starter datasets.
            '102' will download the main dataset with 102 features,
            around 110 GB and 38k grasp attempts.
            See https://sites.google.com/site/brainrobotdata/home
            for a full listing.

        # Returns:

            (features_complete_list, attempt_count)

            features_complete_list: is a list of all feature strings in the
                fixedLengthFeatureDict and sequenceFeatureDict.
            attempt_count: total number of grasp attempts
        """
        dataset = self._update_dataset_param(dataset)
        csv_files = self._get_feature_csv_file_paths(dataset)
        # print('csvfiles_length:', len(csv_files))
        features_complete_list, _, feature_count, attempt_count = self._get_grasp_tfrecord_info(csv_files[-1])

        return features_complete_list, attempt_count

    def _get_tfrecord_path_glob_pattern(self, dataset=None):
        """Get the Glob string pattern for matching the specified dataset tfrecords.

        This will often be used in conjunction with the RecordInput class if you need
        a custom dataset loading function.

        # Arguments
            data_dir: The path to the folder containing the grasp dataset.

            dataset: The name of the dataset to download, downloads all by default
                with the '' parameter, 102 will download the 102 feature dataset
                found in grasp_listing.txt.
        """
        dataset = self._update_dataset_param(dataset)
        return os.path.join(os.path.expanduser(self.data_dir), '*{}.tfrecord*'.format(dataset))

    def _get_grasp_tfrecord_info(self, feature_csv_file):
        """
        # Arguments

            feature_csv_file: path to the feature csv file for this dataset

        # Returns
            features: complete list of all features for this dataset aka tfrecord group
            tfrecord_paths: paths to all tfrecords for this dataset
            feature_count: total number of features
            attempt_count: total number of grasp attempts
        """
        features = np.genfromtxt(os.path.join(os.path.expanduser(self.data_dir), feature_csv_file), dtype=str)
        feature_count = int(features[0])
        attempt_count = int(features[1])
        features = features[2:]
        # Workaround for csv files which may not actually list the key features below,
        # although they have been added to the dataset itself.
        if not any('grasp_success' in s for s in features):
            features = np.append(features, 'grasp_success')
            feature_count += 1
        if not any('gripper/status' in s for s in features):
            features = np.append(features, 'gripper/status')
            feature_count += 1
        # print('_get_grasp_tfrecord_info::feature_complete_list:', features)
        # note that the tfrecords are often named '*{}.tfrecord*-of-*'
        tfrecord_paths = gfile.Glob(self._get_tfrecord_path_glob_pattern())
        return features, tfrecord_paths, feature_count, attempt_count

    @staticmethod
    def get_time_ordered_features(features, feature_type='all', step='all', record_type='fixed'):
        """Get list of all image features ordered by time, features are identified by a string path.
            See https://docs.google.com/spreadsheets/d/1GiPt57nCCbA_2EVkFTtf49Q9qeDWgmgEQ6DhxxAIIao/edit#gid=0
            for mostly correct details.
            These slides also help make the order much more clear:
            https://docs.google.com/presentation/d/13RdgkZQ_neqeXwYU3at2fP4RLl4qb1r74CTehq8d_Qc/edit?usp=sharing
            Also see the downloaded grasp_listing.txt and feature_*.csv files that the downloader puts
            into ~/.keras/datasets/grasping by default.

            TODO(ahundt) make default use 'all' for every param, and ensure that order is correct chronologically.
        # Arguments

            features: list of feature TFRecord strings
            step: string indicating which parts of the grasp sequence should be returned.
                'all' retrieves data from all steps of the grasp sequence,
                'camera' get camera intrinsics matrix and/or camera_T_base transform,
                'robot' get robot name.
                Remaining options in chronological order are:
                'view_clear_scene' shows the clear scene before the arm enters the camera view,
                'move_to_grasp' are the up to 10 steps when the gripper is moving towards an object
                    it will try to grasp, also known as the grasp phase.
                'close_gripper' when the gripper is actually closed.
                 'test_success' with the pre and post drop images,

            feature_type: feature data type, 'all' matches all types, or one of:
                'transforms/base_T_endeffector/vec_quat_7'
                    A pose is a 6 degree of freedom rigid transform represented with 7 values:
                       vector (x, y, z) and quaternion (x, y, z, w).
                       A pose is always annotated with the target and source frames of reference.
                       For example, base_T_camera is a transform that takes a point in the camera frame
                       of reference and transforms it to the base frame of reference.
                'camera/transforms/camera_T_base/matrix44'
                    Same as base_T_endeffector but from the camera center to the robot base
                'camera/intrinsics/matrix33'
                    The 3x3 camera intrinsics matrix.
                'commanded_pose'
                    Commanded pose is the input to the inverse kinematics computation,
                    which is used to determine desired joint positions.
                'reached_pose'
                    Pose calculated by forward kinematics from current robot state in status updates.
                'joint/commanded_torques'
                    Commanded torques are the torque values calculated from the inverse kinematics
                    of the commanded pose and the robot driver.
                'joint/external_torques'
                    External torques are torques at each joint that are a result of gravity and
                    other external forces. These values are reported by the robot status query.
                'joint/positions'
                    Robot joint positions as reported by the robot status query.
                'joint/velocities'
                    Robot joint velocities as reported by the robot status query.
                'depth_image/encoded'
                    Depth image is encoded as an RGB PNG image where the RGB 24-bit value is an
                    integer depth with scale 1/256 of a millimeter. See `depth_image_encoding.py`.
                'depth_image/decoded'
                    This feature is calculated at runtime if `_image_decode()` is called.
                    Uncompressed 'depth_image/encoded'.
                    There may also be uncompressed ops 'depth_image/decoded'
                '/image/encoded'
                    Camera RGB images are stored in JPEG format, be careful not to mismatch on
                    depth_image/encoded, so include the leading slash.
                    There may also be uncompressed ops '/image/decoded'
                    if `_image_decode()` is called.
                '/image/decoded'
                    This feature is calculated at runtime if `_image_decode()` is called.
                    Uncompressed '/image/encoded'.
                    There may also be uncompressed ops '/image/decoded'
                'params'
                    This is a simplified representation of a commanded robot pose and gripper status.
                    These are the values that were solved for in the network, the output of
                    the Cross Entropy Method.
                'name'
                    A string indicating the name of the robot
                'status'
                    'gripper/status' is a 0 to 1 value which indicates how open or closed the
                    gripper was at the end of a grasp attempt.
                'grasp_success'
                    1 if the gripper is determined to have successfully grasped an object on this attempt,
                    0 otherwise. This incorporates a combination of 'gripper/status', the open closed angle
                    of the gripper as the dominant indicator, with image differencing between the final
                    two object drop image features as a secondary indicator of grasp success.

            record_type:
                'all' will match any feature type.
                'fixed' matches fixed length record types (most records, use this if unsure)
                'sequence' matches sequence records (ongoing variable length updates of robot pose)
                Used to separate sequence data types from non sequence data types, since they can be
                variable length and need to be handled separately. In particular this separates out
                variable length vectors of poses.

                This is because data sequences must be read separately from fixed length features
                see `tf.parse_single_sequence_example()` which can do both but they must be defined separately and
                `tf.parse_single_example()` which can only do fixed length features

        # Returns
            TODO(ahundt) may just be returning a list of feature name strings, no tuples at all.
            tuple of size two containing:
            list of fixed length features organized by time step in a single grasp and
            list of sequence features organized by time step in a single grasp
        """
        matching_features = []

        def _match_feature(features, feature_name_regex, feature_type='', exclude_substring=None, exclude_regex=None):
            """Get first feature from the list that meets requirements.
            Used to ensure correct ordering and easy selection of features.
            some silly tricks to get the feature names in the right order while
            allowing variation between the various datasets.
            For example, we need to make sure 'grasp/image/encoded comes'
            before grasp/0/* and post_grasp/*, but with a simple substring
            match 'grasp/', all of the above will be matched in the wrong order.
            `r'/\\d+/'` (single backslash) regex will exclude /0/, /1/, and /10/.
            """
            if ('image/encoded' in feature_type) and ('depth' not in feature_type):
                feature_type = '/image/encoded'
            for ifname in features:
                if (bool(re.search(feature_name_regex, ifname)) and  # feature name matches
                        ((exclude_substring is None) or (exclude_substring not in ifname)) and
                        ((exclude_regex is None) or not bool(re.search(exclude_regex, ifname))) and
                        (feature_type in ifname)):
                    return [str(ifname)]
            return []

        # see feature type docstring for details
        if record_type is 'fixed':
            exclude_substring = 'sequence'
        else:
            exclude_substring = None

        if step in ['camera', 'all', '']:
            # the r'^ in r'^camera/' makes sure the start of the string matches
            matching_features.extend(_match_feature(features, r'^camera/', feature_type))
        if step in ['robot', 'all', '']:
            matching_features.extend(_match_feature(features, r'^robot/', feature_type))
        # r'/\d/' is a regex that will exclude things like /0/, /1/, through /10/
        # this is the first pre grasp image, even though it is called grasp
        if step in ['view_clear_scene', 'all', '']:
            matching_features.extend(_match_feature(features, r'^initial/', feature_type))
            matching_features.extend(_match_feature(features, r'^approach/', feature_type, exclude_substring))
            matching_features.extend(_match_feature(features, r'^approach_sequence/', feature_type, exclude_substring))
            matching_features.extend(_match_feature(features, r'^pregrasp/', feature_type, 'post', r'/\d+/'))
            matching_features.extend(_match_feature(features, r'^grasp/', feature_type, 'post', r'/\d+/'))

        # up to 11 grasp steps in the datasets
        if step in ['move_to_grasp', 'all', '']:
            max_grasp_steps = 11  # 0 through 10
            for i in range(max_grasp_steps):
                matching_features.extend(_match_feature(features, r'^grasp/{}/'.format(i), feature_type, 'post'))
        # closing the gripper
        if step in ['close_gripper', 'all', '']:
            matching_features.extend(_match_feature(features, r'^gripper/', feature_type, exclude_substring))
            # Not totally sure, but probably the transforms and angles from as the gripper is closing (might shift a bit)
            matching_features.extend(_match_feature(features, r'^gripper_sequence/', feature_type, exclude_substring))
            # Withdraw the closed gripper from the bin vertically upward, roughly 10 cm above the bin surface.
            matching_features.extend(_match_feature(features, r'^withdraw_sequence/', feature_type, exclude_substring))
            matching_features.extend(_match_feature(features, r'^post_grasp/', feature_type))

        if step in ['test_success', 'all', '']:
            matching_features.extend(_match_feature(features, r'^present/', feature_type, exclude_substring))
            matching_features.extend(_match_feature(features, r'^present_sequence/', feature_type, exclude_substring))
            # After presentation (or after withdraw, if present is skipped),
            # the object is moved to a random position about 10 cm above the bin to drop the object.
            matching_features.extend(_match_feature(features, r'^raise/', feature_type))
            # Open the gripper and drop the object (if any object is held) into the bin.
            matching_features.extend(_match_feature(features, r'^drop/', feature_type))
            # Images recorded after withdraw, raise, and the drop.
            matching_features.extend(_match_feature(features, r'^post_drop/', feature_type))
            matching_features.extend(_match_feature(features, r'^grasp_success', feature_type))
        return matching_features

    @staticmethod
    def parse_grasp_attempt_protobuf(serialized_grasp_attempt_proto, features_complete_list):
        """Parses an Example protobuf containing a training example of an image.
        The output of the build_image_data.py image preprocessing script is a dataset
        containing serialized Example protocol buffers. Each Example proto contains
        the fields described in get_time_ordered_features.
        Args:
            example_serialized: scalar Tensor tf.string containing a serialized
            Example protocol buffer.

            list of tensors corresponding to images, actions, and states. Each images
            tensor is 4D, batch x height x width x channels. The state and
            action tensors are 3D, batch x time x dimension.
        Returns:
            feature_op_dict: dictionary of Tensors for every feature,
                use `get_time_ordered_features` to select the subsets you need.
            sequence_feature_op_dict: dictionary of sequence tensors for every features,
                contains base to end effector transforms.
        """
        # Dense features in Example proto.
        num_grasp_steps_name = 'num_grasp_steps'
        camera_to_base_name = 'camera/transforms/camera_T_base/matrix44'
        camera_intrinsics_name = 'camera/intrinsics/matrix33'
        grasp_success_name = 'grasp_success'
        # TODO(ahundt) make sure gripper/status is in the right place and not handled twice
        gripper_status = 'gripper/status'

        # setup one time features like the camera and number of grasp steps
        features_dict = {
            num_grasp_steps_name: tf.FixedLenFeature([1], tf.string),
            camera_to_base_name: tf.FixedLenFeature([4, 4], tf.float32),
            camera_intrinsics_name: tf.FixedLenFeature([3, 3], tf.float32),
            grasp_success_name: tf.FixedLenFeature([1], tf.float32),
            gripper_status: tf.FixedLenFeature([1], tf.float32),
        }

        # load all the images
        ordered_image_feature_names = GraspDataset.get_time_ordered_features(
            features_complete_list,
            feature_type='/image/encoded')
        features_dict.update({image_name: tf.FixedLenFeature([1], tf.string)
                              for image_name in ordered_image_feature_names})

        # load all the depth images
        ordered_depth_feature_names = GraspDataset.get_time_ordered_features(
            features_complete_list,
            feature_type='depth_image')
        features_dict.update({image_name: tf.FixedLenFeature([1], tf.string)
                              for image_name in ordered_depth_feature_names})

        # load all vec/quat base to end effector transforms that aren't sequences
        base_to_endeffector_names = GraspDataset.get_time_ordered_features(
            features_complete_list,
            feature_type='transforms/base_T_endeffector/vec_quat_7'
            )
        features_dict.update({x_form: tf.FixedLenFeature([7], tf.float32)
                              for x_form in base_to_endeffector_names})

        # load all vec/quat base to end effector transforms that are sequences
        base_to_endeffector_sequence_names = GraspDataset.get_time_ordered_features(
            features_complete_list,
            feature_type='sequence/transforms/base_T_endeffector/vec_quat_7',
            record_type='sequence'  # Don't exclude sequences
            )
        sequence_features_dict = {x_form: tf.FixedLenSequenceFeature([7], tf.float32, allow_missing=True)
                                  for x_form in base_to_endeffector_sequence_names}

        # load transforms that were modified for their network from the original paper
        transform_adapted_for_network = GraspDataset.get_time_ordered_features(
            features_complete_list,
            feature_type='params'
            )
        features_dict.update({x_form: tf.FixedLenFeature([5], tf.float32)
                              for x_form in transform_adapted_for_network})

        # extract all the features from the file, return two dicts
        return tf.parse_single_sequence_example(serialized_grasp_attempt_proto,
                                                context_features=features_dict,
                                                sequence_features=sequence_features_dict)

    def get_simple_parallel_dataset_ops(self, dataset=None, batch_size=1, buffer_size=100, parallelism=10):
        """Simple unordered & parallel TensorFlow ops that go through the whole dataset.

        # Returns

            A list of tuples ([(fixedLengthFeatureDict, sequenceFeatureDict)], features_complete_list, num_samples).
            fixedLengthFeatureDict maps from the feature strings of most features to their TF ops.
            sequenceFeatureDict maps from feature strings to time ordered sequences of poses transforming
            from the robot base to end effector.
            features_complete_list: a list of all feature strings in the fixedLengthFeatureDict and sequenceFeatureDict,
                and a parameter for get_time_ordered_features().
            num_samples: the number of samples in the dataset, used for configuring the size of one training epoch

        """
        tf_glob = self._get_tfrecord_path_glob_pattern(dataset=dataset)
        record_input = data_flow_ops.RecordInput(tf_glob, batch_size, buffer_size, parallelism)
        records_op = record_input.get_yield_op()
        records_op = tf.split(records_op, batch_size, 0)
        records_op = [tf.reshape(record, []) for record in records_op]
        features_complete_list, num_samples = self.get_features()
        feature_op_dicts = [self.parse_grasp_attempt_protobuf(serialized_protobuf, features_complete_list) for serialized_protobuf in records_op]
        # TODO(ahundt) https://www.tensorflow.org/performance/performance_models
        # make sure records are always ready to go
        # staging_area = tf.contrib.staging.StagingArea()
        dict_and_feature_tuple_list = []
        for feature_op_dict, sequence_op_dict in feature_op_dicts:
            features_op_dict, new_feature_list = GraspDataset._image_decode(feature_op_dict)
            dict_and_feature_tuple_list.append((features_op_dict, sequence_op_dict))
        # the new_feature_list should be the same for all the ops
        features_complete_list = np.append(features_complete_list, new_feature_list)

        return dict_and_feature_tuple_list, features_complete_list, num_samples

    def get_simple_tfrecordreader_dataset_ops(self, batch_size=1):
        """Get a dataset reading op from tfrecordreader.
        You will have to call tf.train.batch and tf.train.start_queue_runners(sess), see create_gif.
        TODO(ahundt) this is deprecated, update with Dataset API https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

        # Returns

            A list of tuples [(fixedLengthFeatureDict, sequenceFeatureDict, features_complete_list)].
            fixedLengthFeatureDict maps from the feature strings of most features to their TF ops.
            sequenceFeatureDict maps from feature strings to time ordered sequences of poses transforming
            from the robot base to end effector.
            features_complete_list: is a list of all feature strings in the fixedLengthFeatureDict and sequenceFeatureDict.
            feature_count: total number of features
            attempt_count: total number of grasp attempts

        """
        [feature_csv_file] = self._get_feature_csv_file_paths()
        features_complete_list, tfrecord_paths, feature_count, attempt_count = self._get_grasp_tfrecord_info(feature_csv_file)
        if not tfrecord_paths:
            raise RuntimeError('No tfrecords found for {}.'.format(feature_csv_file))
        filename_queue = tf.train.string_input_producer(tfrecord_paths, shuffle=False)
        # note that this API is deprecated
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        feature_op_dicts = [self.parse_grasp_attempt_protobuf(serialized_example, features_complete_list)]
        # TODO(ahundt) https://www.tensorflow.org/performance/performance_models
        # make sure records are always ready to go
        # staging_area = tf.contrib.staging.StagingArea()
        dict_and_feature_tuple_list = []
        for feature_op_dict, sequence_op_dict in feature_op_dicts:
            features_op_dict, new_feature_list = GraspDataset._image_decode(feature_op_dict)
            dict_and_feature_tuple_list.append((features_op_dict, sequence_op_dict))
        # the new_feature_list should be the same for all the ops
        features_complete_list = np.append(features_complete_list, new_feature_list)

        return dict_and_feature_tuple_list, features_complete_list, feature_count, attempt_count

    @staticmethod
    def _image_decode(feature_op_dict):
        """Add features to dict that supply decoded png and jpeg images for any encoded images present.
        Feature path that is 'image/encoded' will also now have 'image/decoded'

        # Returns

            updated feature_op_dict, new_feature_list
        """
        features = [feature for (feature, tf_op) in iteritems(feature_op_dict)]
        image_features = GraspDataset.get_time_ordered_features(features, '/image/encoded')
        image_features.extend(GraspDataset.get_time_ordered_features(features, 'depth_image/encoded'))
        new_feature_list = []
        for image_feature in image_features:
            image_buffer = tf.reshape(feature_op_dict[image_feature], shape=[])
            if 'depth' in image_feature:
                image = tf.image.decode_png(image_buffer, channels=FLAGS.sensor_color_channels)
            else:
                image = tf.image.decode_jpeg(image_buffer, channels=FLAGS.sensor_color_channels)
            image.set_shape([FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels])
            image = tf.reshape(image, [1, FLAGS.sensor_image_height, FLAGS.sensor_image_width, FLAGS.sensor_color_channels])
            decoded_image_feature = image_feature.replace('encoded', 'decoded')
            feature_op_dict[decoded_image_feature] = image
            new_feature_list.append(decoded_image_feature)

        return feature_op_dict, new_feature_list

    def npy_to_gif(self, npy, filename):
        clip = mpy.ImageSequenceClip(list(npy), fps=2)
        clip.write_gif(filename)

    def create_gif(self, sess):
        """ Create gifs of loaded dataset
        """
        mkdir_p(FLAGS.visualization_dir)
        feature_csv_files = self._get_feature_csv_file_paths()
        for feature_csv_file in feature_csv_files:
            """Create input tfrecord tensors.

            Args:
            novel: whether or not to grab novel or seen images.
            Returns:
            list of tensors corresponding to images. The images
            tensor is 5D, batch x time x height x width x channels.
            Raises:
            RuntimeError: if no files found.
            """
            # decode all the image ops and other features
            [(features_op_dict, _)], features_complete_list, _, attempt_count = self.get_simple_tfrecordreader_dataset_ops()
            print(features_complete_list)
            ordered_image_feature_names = GraspDataset.get_time_ordered_features(features_complete_list, '/image/decoded')

            grasp_success_feature_name = GraspDataset.get_time_ordered_features(
                features_complete_list,
                feature_type='grasp_success'
            )
            # should only be a list of length 1, just make into a single string
            grasp_success_feature_name = grasp_success_feature_name[0]

            image_seq = []
            for image_name in ordered_image_feature_names:
                image_seq.append(features_op_dict[image_name])

            image_seq = tf.concat(image_seq, 0)
            # output won't be correct now if batch size is anything but 1
            batch_size = 1
            print('fo_dict: ', features_op_dict)
            print('fodgs: ', features_op_dict[grasp_success_feature_name])

            train_image_dict = tf.train.batch(
                {'image_seq': image_seq,
                 grasp_success_feature_name: features_op_dict[grasp_success_feature_name]},
                batch_size,
                num_threads=1,
                capacity=1)
            tf.train.start_queue_runners(sess)
            sess.run(tf.global_variables_initializer())

            # note: if batch size doesn't divide evenly some items may not be output correctly
            for attempt_num in range(attempt_count / batch_size):
                numpy_data_dict = sess.run(train_image_dict)
                for i in range(batch_size):
                    video = numpy_data_dict['image_seq'][i]
                    gif_filename = (os.path.basename(feature_csv_file)[:-4] + '_grasp_' + str(int(attempt_num)) +
                                    '_success_' + str(int(numpy_data_dict[grasp_success_feature_name])) + '.gif')
                    gif_path = os.path.join(FLAGS.visualization_dir, gif_filename)
                    self.npy_to_gif(video, gif_path)


if __name__ == '__main__':
    with tf.Session() as sess:
        gd = GraspDataset()
        gd.create_gif(sess)

