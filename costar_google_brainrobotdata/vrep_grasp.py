# -*- coding: utf-8 -*-
"""Code for visualizing the grasp attempt examples."""
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
from ply import write_xyz_rgb_as_ply

import moviepy.editor as mpy
from grasp_dataset import GraspDataset
import grasp_geometry
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
tf.flags.DEFINE_string('vrepDebugMode', 'save_ply', """Options are: '', 'fixed_depth', 'save_ply'.""")

flags.FLAGS._parse_flags()
FLAGS = flags.FLAGS


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
            # display the reply from V-REP (in this case, the handle of the created dummy)
            print ('Dummy name:', display_name, ' handle: ', ret_ints[0], ' transform: ', transform)
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

    def visualize(self, tf_session, dataset=FLAGS.grasp_dataset, batch_size=1, parent_name='LBR_iiwa_14_R820'):
        """Visualize one dataset in V-REP
        """
        grasp_dataset_object = GraspDataset(dataset=dataset)
        batch_size = 1
        feature_op_dicts, features_complete_list, num_samples = grasp_dataset_object._get_simple_parallel_dataset_ops(
            batch_size=batch_size)

        tf_session.run(tf.global_variables_initializer())

        error_code, parent_handle = v.simxGetObjectHandle(self.client_id, parent_name, v.simx_opmode_blocking)
        if error_code is -1:
            parent_handle = -1
            print 'could not find object with the specified name, so putting objects in world frame:', parent_name

        for attempt_num in range(num_samples / batch_size):
            output_features_dict = tf_session.run(feature_op_dicts)
            for features_dict_np, sequence_dict_np in output_features_dict:
                # TODO(ahundt) actually put transforms into V-REP or pybullet
                self._visualize_one_grasp_attempt(
                        grasp_dataset_object, features_complete_list, features_dict_np, parent_handle,
                        dataset_name=dataset,
                        attempt_num=attempt_num)

                # returnCode, dummyHandle = v.simxCreateDummy(self.client_id, 0.1, colors=None, operationMode=simx_opmode_blocking)

    def _visualize_one_grasp_attempt(self, grasp_dataset_object, features_complete_list, features_dict_np, parent_handle,
                                     dataset_name=FLAGS.grasp_dataset,
                                     attempt_num=0,
                                     grasp_sequence_min_time_step=FLAGS.grasp_sequence_min_time_step,
                                     grasp_sequence_max_time_step=FLAGS.grasp_sequence_max_time_step,
                                     visualization_dir=FLAGS.visualization_dir,
                                     vrepDebugMode=FLAGS.vrepDebugMode):
        """Take an extracted grasp attempt tfrecord numpy dictionary and visualize it in vrep

        # Params

        parent_handle: the frame in which to display transforms, defaults to base frame of 'LBR_iiwa_14_R820'
        """
        # TODO(ahundt) actually put transforms into V-REP or pybullet
        base_to_endeffector_transforms = grasp_dataset_object.get_time_ordered_features(
            features_complete_list,
            feature_type='transforms/base_T_endeffector/vec_quat_7')
        camera_to_base_transform_name = 'camera/transforms/camera_T_base/matrix44'
        camera_intrinsics_name = 'camera/intrinsics/matrix33'

        depth_image_features = grasp_dataset_object.get_time_ordered_features(
            features_complete_list,
            feature_type='depth_image/decoded'
        )

        rgb_image_features = grasp_dataset_object.get_time_ordered_features(
            features_complete_list,
            feature_type='/image/decoded'
        )

        grasp_success_feature_name = grasp_dataset_object.get_time_ordered_features(
            features_complete_list,
            feature_type='grasp_success'
        )[0]

        camera_intrinsics_matrix = features_dict_np[camera_intrinsics_name]
        camera_to_base_4x4matrix = features_dict_np[camera_to_base_transform_name]
        base_to_camera_vec_quat_7 = grasp_geometry.matrix_to_vector_quaternion_array(camera_to_base_4x4matrix)
        camera_T_base_handle = self.create_dummy('camera_T_base', base_to_camera_vec_quat_7, parent_handle)
        # gripper_positions = [features_dict_np[transform_name] for transform_name in base_to_endeffector_transforms]
        for i, transform_name, depth_name, rgb_name in zip(range(len(base_to_endeffector_transforms)),
                                                           base_to_endeffector_transforms,
                                                           depth_image_features, rgb_image_features):
            # 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
            empty_buffer = bytearray()
            # 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
            gripper_pose = features_dict_np[transform_name]
            camera_T_endeffector_ptrans, base_T_endeffector_ptrans, camera_T_base_ptrans = grasp_geometry.grasp_dataset_to_ptransform(
                camera_to_base_4x4matrix,
                gripper_pose
            )

            base_T_endeffector_vec_quat = grasp_geometry.ptransform_to_vector_quaternion_array(base_T_endeffector_ptrans)
            bTe_display_name = str(i).zfill(2) + '_base_T_endeffector_ptransform_initial_commanded'
            self.create_dummy(bTe_display_name, base_T_endeffector_vec_quat, parent_handle)
            assert(np.allclose(base_T_endeffector_vec_quat, gripper_pose))

            # verify that another transform path gets the same result
            test_base_to_camera_vec_quat_7 = grasp_geometry.ptransform_to_vector_quaternion_array(camera_T_base_ptrans)
            assert(np.allclose(base_to_camera_vec_quat_7, base_to_camera_vec_quat_7))

            cTe_display_name = str(i).zfill(2) + '_camera_T_endeffector'
            cTe_vec_quat = grasp_geometry.ptransform_to_vector_quaternion_array(camera_T_endeffector_ptrans)
            self.create_dummy(cTe_display_name, cTe_vec_quat, camera_T_base_handle)

            if i == 0:
                clear_frame_depth_image = np.squeeze(features_dict_np[depth_name])
            # format the dummy string nicely for display
            transform_display_name = str(i).zfill(2) + '_' + transform_name.replace(
                '/transforms/base_T_endeffector/vec_quat_7', '').replace('/', '_')
            print transform_name, transform_display_name, gripper_pose
            # display the gripper pose
            self.create_dummy(transform_display_name, gripper_pose, parent_handle)

            ee_cloud_point, ee_image_coordinate = grasp_geometry.endeffector_image_coordinate_and_cloud_point(
                clear_frame_depth_image, camera_intrinsics_matrix, camera_T_endeffector_ptrans)

            # Create a dummy for the key depth point and display it
            depth_point_dummy_ptrans = grasp_geometry.vector_to_ptransform(ee_cloud_point)
            depth_point_display_name = str(i).zfill(2) + '_depth_point'
            depth_point_vec_quat = grasp_geometry.ptransform_to_vector_quaternion_array(depth_point_dummy_ptrans)
            depth_point_dummy_handle = self.create_dummy(depth_point_display_name, depth_point_vec_quat, camera_T_base_handle)

            # get the transform for the gripper relative to the key depth point and display it
            # it should coincide with the gripper pose if done correctly
            surface_relative_transform_vec_quat = grasp_geometry.surface_relative_transform(
                clear_frame_depth_image, camera_intrinsics_matrix, camera_T_endeffector_ptrans)
            surface_relative_transform_display_name = str(i).zfill(2) + '_depth_point'
            surface_relative_transform_dummy_handle = self.create_dummy(surface_relative_transform_display_name,
                                                                        surface_relative_transform_vec_quat,
                                                                        depth_point_dummy_handle)

            rgb_image = features_dict_np[rgb_name]
            print rgb_name, rgb_image.shape, rgb_image
            # TODO(ahundt) move depth image creation into tensorflow ops
            # TODO(ahundt) check scale
            # TODO(ahundt) move squeeze steps into dataset api if possible
            depth_image_float_format = np.squeeze(features_dict_np[depth_name])
            if np.count_nonzero(depth_image_float_format) is 0:
                print 'WARNING: DEPTH IMAGE IS ALL ZEROS'
            print depth_name, depth_image_float_format.shape, depth_image_float_format
            if ((grasp_sequence_min_time_step is None or i >= grasp_sequence_min_time_step) and
                    (grasp_sequence_max_time_step is None or i <= grasp_sequence_max_time_step)):
                # only output one depth image while debugging
                # mp.pyplot.imshow(depth_image_float_format, block=True)
                print 'plot done'
                # TODO(ahundt) uncomment next line after debugging is done
                point_cloud = depth_image_to_point_cloud(depth_image_float_format, camera_intrinsics_matrix)
                if 'fixed_depth' in vrepDebugMode:
                    point_cloud = depth_image_to_point_cloud(np.ones(depth_image_float_format.shape), camera_intrinsics_matrix)
                print 'point_cloud.shape:', point_cloud.shape, 'rgb_image.shape:', rgb_image.shape
                point_cloud_display_name = ('point_cloud_' + str(dataset_name) + '_' + str(attempt_num) + '_' + str(i).zfill(2) +
                                            '_rgbd_' + depth_name.replace('/depth_image/decoded', '').replace('/', '_') +
                                            '_success_' + str(int(features_dict_np[grasp_success_feature_name])))
                print 'point_cloud:', point_cloud.transpose()[:30, :3]
                path = os.path.join(visualization_dir, point_cloud_display_name + '.ply')
                print 'point_cloud.size:', point_cloud.size
                xyz = point_cloud.reshape([point_cloud.size/3, 3])
                rgb = np.squeeze(rgb_image).reshape([point_cloud.size/3, 3])
                if 'save_ply' in vrepDebugMode:
                    write_xyz_rgb_as_ply(xyz, rgb, path)
                xyz = xyz[:3000, :]
                # xyz = np.array([[0,0,0], [0,0,1], [0,1,0], [1,0,0]])
                self.create_point_cloud(point_cloud_display_name, xyz, base_to_camera_vec_quat_7, rgb, parent_handle)

    def __del__(self):
        v.simxFinish(-1)


if __name__ == '__main__':

    with tf.Session() as sess:
        sim = VREPGraspSimulation()
        sim.visualize(sess)

