# -*- coding: utf-8 -*-
"""Code for building the input for the prediction model."""
# from __future__ import unicode_literals
import os
import errno

import numpy as np

try:
    import vrep.vrep as v
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in PYTHONPATH folder relative to this file,')
    print ('or appropriately adjust the file "vrep.py. Also follow the"')
    print ('ReadMe.txt in the vrep remote API folder')
    print ('--------------------------------------------------------------')
    print ('')

import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.ops import data_flow_ops
from keras.utils import get_file

import moviepy.editor as mpy
from grasp_dataset import GraspDataset
from depth_image_encoding import ImageToFloatArray
from depth_image_encoding import depth_image_to_point_cloud

# https://github.com/jrl-umi3218/Eigen3ToPython/tree/topic/Cython
# alternatives to consider:
# https://github.com/adamlwgriffiths/Pyrr
# https://github.com/KieranWynn/pyquaternion
import eigen as e
# import matplotlib as mp

tf.flags.DEFINE_string('vrepConnectionAddress', '127.0.0.1', 'The IP address of the running V-REP simulation.')
tf.flags.DEFINE_integer('vrepConnectionPort', 19999, 'ip port for connecting to V-REP')
tf.flags.DEFINE_boolean('vrepWaitUntilConnected', True, 'block startup call until vrep is connected')
tf.flags.DEFINE_boolean('vrepDoNotReconnectOnceDisconnected', True, '')
tf.flags.DEFINE_integer('vrepTimeOutInMs', 5000, 'Timeout in milliseconds upon which connection fails')
tf.flags.DEFINE_integer('vrepCommThreadCycleInMs', 5, 'time between communication cycles')

FLAGS = flags.FLAGS


def matrix_to_vector_quaternion_array(matrix, inverse=False):
    """Convert a 4x4 Rt transformation matrix into an vector quaternion array
    containing 3 vector entries (x, y, z) and 4 quaternion entries (x, y, z, w)
    """
    rot = e.Matrix3d(matrix[:3, :3])
    quaternion = e.Quaterniond(rot)
    translation = matrix[:3, 3].transpose()
    if inverse:
        quaternion = quaternion.inverse()
        translation *= -1
    q_floats_array = np.array(quaternion.coeffs()).astype(np.float32)
    vec_quat_7 = np.append(translation, q_floats_array)
    print vec_quat_7
    return vec_quat_7


class VREPGraspSimulation(object):

    def __init__(self):
        """Start the connection to the remote V-REP simulation
        """
        print 'Program started'
        # just in case, close all opened connections
        v.simxFinish(-1)
        # Connect to V-REP
        self.client_id = v.simxStart(FLAGS.vrepConnectionAddress,
                                     FLAGS.vrepConnectionPort,
                                     FLAGS.vrepWaitUntilConnected,
                                     FLAGS.vrepDoNotReconnectOnceDisconnected,
                                     FLAGS.vrepTimeOutInMs,
                                     FLAGS.vrepCommThreadCycleInMs)
        if self.client_id != -1:
            print 'Connected to remote API server'
        return

    def create_dummy(self, display_name, transform, parent_handle=-1):
        """Create a dummy object in the simulation

        # Arguments

            transform_display_name: name string to use for the object in the vrep scene
            transform: 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
            parent_handle: -1 is the world frame, any other int should be a vrep object handle
        """
        # 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
        empty_buffer = bytearray()
        res, ret_ints, ret_floats, ret_strings, ret_buffer = v.simxCallScriptFunction(
            self.client_id,
            'remoteApiCommandServer',
            v.sim_scripttype_childscript,
            'createDummy_function',
            [parent_handle],
            transform,
            [display_name],
            empty_buffer,
            v.simx_opmode_blocking)
        if res == v.simx_return_ok:
            print ('Dummy handle: ', ret_ints[0])  # display the reply from V-REP (in this case, the handle of the created dummy)
        else:
            print 'create_dummy remote function call failed.'
            return -1
        return ret_ints[0]

    def create_point_cloud(self, display_name, points, transform, color_image=None, parent_handle=-1):
        """Create a dummy object in the simulation

        # Arguments

            transform_display_name: name string to use for the object in the vrep scene
            transform: 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
            parent_handle: -1 is the world frame, any other int should be a vrep object handle
        """
        # 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
        empty_buffer = bytearray()
        # strings = [display_name, unicode(points.flatten().tostring(), 'utf-8')]
        # for when the numpy array can be packed in a byte string
        # strings = [display_name, points.flatten().tostring()]
        strings = [display_name]
        # if color_image is not None:
        #     # strings.extend(unicode(color_image.flatten().tostring(), 'utf-8'))
        #     # for when the numpy array can be packed in a byte string
        #     trings.extend(color_image.flatten().tostring())
        transform_entries = 7
        print 'points.size:', points.size
        res, ret_ints, ret_floats, ret_strings, ret_buffer = v.simxCallScriptFunction(
            self.client_id,
            'remoteApiCommandServer',
            v.sim_scripttype_childscript,
            'createPointCloud_function',
            [parent_handle, transform_entries, points.size],
            np.append(transform, points),
            # for when the double array can be packed in a byte string
            # transform,
            strings,
            empty_buffer,
            v.simx_opmode_blocking)
        if res == v.simx_return_ok:
            print ('point cloud handle: ', ret_ints[0])  # display the reply from V-REP (in this case, the handle of the created dummy)
            return ret_ints[0]
        else:
            print 'create_point_cloud remote function call failed.'
            return -1

    def visualize(self, tf_session, dataset=None, batch_size=1, parent_name='LBR_iiwa_14_R820'):
        """Visualize one dataset in V-REP
        """
        grasp_dataset = GraspDataset(dataset)
        feature_op_dicts, features_complete_list = grasp_dataset.get_simple_parallel_dataset_ops()
        # TODO(ahundt) https://www.tensorflow.org/performance/performance_models
        # make sure records are always ready to go
        # staging_area = tf.contrib.staging.StagingArea()

        tf_session.run(tf.global_variables_initializer())
        output_features_dict = tf_session.run(feature_op_dicts)

        error_code, parent_handle = v.simxGetObjectHandle(self.client_id, parent_name, v.simx_opmode_blocking)
        if error_code is -1:
            parent_handle = -1
            print 'could not find object with the specified name, so putting objects in world frame:', parent_name

        for features_dict_np, sequence_dict_np in output_features_dict:
            # TODO(ahundt) actually put transforms into V-REP or pybullet
            camera_to_base_transform, camera_intrinsics = self._visualize_one_grasp_attempt(
                    grasp_dataset, features_complete_list, features_dict_np, parent_handle)

            # returnCode, dummyHandle = v.simxCreateDummy(self.client_id, 0.1, colors=None, operationMode=simx_opmode_blocking)

    def _visualize_one_grasp_attempt(self, grasp_dataset, features_complete_list, features_dict_np, parent_handle):
        """Take an extracted grasp attempt tfrecord numpy dictionary and visualize it in vrep
        """
        # TODO(ahundt) actually put transforms into V-REP or pybullet
        base_to_endeffector_transforms = grasp_dataset.get_time_ordered_features(
            features_complete_list,
            feature_type='transforms/base_T_endeffector/vec_quat_7')
        camera_to_base_transform_name = 'camera/transforms/camera_T_base/matrix44'
        camera_intrinsics_name = 'camera/intrinsics/matrix33'

        depth_image_features = grasp_dataset.get_time_ordered_features(
            features_complete_list,
            feature_type='depth_image/decoded'
        )

        rgb_image_features = grasp_dataset.get_time_ordered_features(
            features_complete_list,
            feature_type='/image/decoded'
        )

        camera_intrinsics_matrix = features_dict_np[camera_intrinsics_name]
        camera_to_base_4x4matrix = features_dict_np[camera_to_base_transform_name]
        base_to_camera_vec_quat_7 = matrix_to_vector_quaternion_array(camera_to_base_4x4matrix)
        # gripper_positions = [features_dict_np[transform_name] for transform_name in base_to_endeffector_transforms]
        for i, transform_name, depth_name, rgb_name in zip(range(len(base_to_endeffector_transforms)), base_to_endeffector_transforms, depth_image_features, rgb_image_features):
            # 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
            empty_buffer = bytearray()
            # 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
            gripper_pose = features_dict_np[transform_name]
            # format the dummy string nicely for display
            transform_display_name = str(i).zfill(2) + '_' + transform_name.replace('/transforms/base_T_endeffector/vec_quat_7', '').replace('/', '_')
            print transform_name, transform_display_name, gripper_pose
            self.create_dummy(transform_display_name, gripper_pose, parent_handle)

            rgb_image = features_dict_np[rgb_name]
            print rgb_name, rgb_image.shape, rgb_image
            # TODO(ahundt) move depth image creation into tensorflow ops
            # TODO(ahundt) check scale
            depth_image_float_format = ImageToFloatArray(np.squeeze(features_dict_np[depth_name]))
            if np.count_nonzero(depth_image_float_format) is 0:
                print 'WARNING: DEPTH IMAGE IS ALL ZEROS'
            print depth_name, depth_image_float_format.shape, depth_image_float_format
            # mp.pyplot.imshow(depth_image_float_format, block=True)
            print 'plot done'
            point_cloud = depth_image_to_point_cloud(depth_image_float_format, camera_intrinsics_matrix)
            point_cloud_display_name = str(i).zfill(2) + '_rgbd_' + depth_name.replace('/depth_image/decoded', '').replace('/', '_')

            self.create_point_cloud(point_cloud_display_name, point_cloud, base_to_camera_vec_quat_7, rgb_image, parent_handle)

    def __del__(self):
        v.simxFinish(-1)


if __name__ == '__main__':

    with tf.Session() as sess:
        sim = VREPGraspSimulation()
        sim.visualize(sess)

