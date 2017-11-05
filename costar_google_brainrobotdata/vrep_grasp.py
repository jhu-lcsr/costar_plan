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
from PIL import Image

import moviepy.editor as mpy
from grasp_dataset import GraspDataset
import grasp_geometry
from depth_image_encoding import depth_image_to_point_cloud

# https://github.com/jrl-umi3218/Eigen3ToPython/
# alternatives to consider:
# https://github.com/adamlwgriffiths/Pyrr
# https://github.com/KieranWynn/pyquaternion
import eigen as e  # https://github.com/jrl-umi3218/Eigen3ToPython
import sva  # https://github.com/jrl-umi3218/SpaceVecAlg
# import matplotlib as mp

tf.flags.DEFINE_string('vrepConnectionAddress', '127.0.0.1', 'The IP address of the running V-REP simulation.')
tf.flags.DEFINE_integer('vrepConnectionPort', 19999, 'ip port for connecting to V-REP')
tf.flags.DEFINE_boolean('vrepWaitUntilConnected', True, 'block startup call until vrep is connected')
tf.flags.DEFINE_boolean('vrepDoNotReconnectOnceDisconnected', True, '')
tf.flags.DEFINE_integer('vrepTimeOutInMs', 5000, 'Timeout in milliseconds upon which connection fails')
tf.flags.DEFINE_integer('vrepCommThreadCycleInMs', 5, 'time between communication cycles')
tf.flags.DEFINE_integer('vrepVisualizeGraspAttempt_min', 0, 'min grasp attempt to display from dataset, or -1 for no limit')
tf.flags.DEFINE_integer('vrepVisualizeGraspAttempt_max', 1, 'max grasp attempt to display from dataset, exclusive, or -1 for no limit')
tf.flags.DEFINE_string('vrepDebugMode', 'save_ply', """Options are: '', 'fixed_depth', 'save_ply'.""")
tf.flags.DEFINE_boolean('vrepVisualizeRGBD', True, 'display the rgbd images and point cloud')
tf.flags.DEFINE_integer('vrepVisualizeRGBD_min', 0, 'min time step on each grasp attempt to display')
tf.flags.DEFINE_integer('vrepVisualizeRGBD_max', 1, 'max time step on each grasp attempt to display, exclusive')
tf.flags.DEFINE_boolean('vrepVisualizeSurfaceRelativeTransform', False, 'display the surface relative transform frames')
tf.flags.DEFINE_string('vrepParentName', 'LBR_iiwa_14_R820', 'The default parent frame name from which to base all visualized transforms.')

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

    def setPose(self, display_name, transform, parent_handle=-1):
        """Set the pose of an object in the simulation

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
            print ('SetPose object name:', display_name, ' handle: ', ret_ints[0], ' transform: ', transform)
        else:
            print 'setPose remote function call failed.'
            return -1
        return ret_ints[0]

    def create_point_cloud(self, display_name, points, transform, color_image=None, parent_handle=-1, clear=True):
        """Create a dummy object in the simulation

        # Arguments

            transform_display_name: name string to use for the object in the vrep scene
            transform: 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
            parent_handle: -1 is the world frame, any other int should be a vrep object handle
            clear: clear the point cloud if it already exists with the provided display name
        """
        # 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
        empty_buffer = bytearray()
        # strings = [display_name, unicode(points.flatten().tostring(), 'utf-8')]
        # for when the numpy array can be packed in a byte string
        # strings = [display_name, points.flatten().tostring()]
        strings = [display_name]
        #     # for when the numpy array can be packed in a byte string
        #     strings.extend(color_image.flatten().tostring())
        transform_entries = 7
        print 'points.size:', points.size
        if clear is True:
            clear = 1
        else:
            clear = 0

        cloud_handle = -1

        # Create the point cloud if it does not exist, or retrieve the handle if it does
        res, ret_ints, ret_floats, ret_strings, ret_buffer = v.simxCallScriptFunction(
            self.client_id,
            'remoteApiCommandServer',
            v.sim_scripttype_childscript,
            'createPointCloud_function',
            [parent_handle, transform_entries, points.size, cloud_handle, clear],
            # np.append(transform, points),
            [],
            # for when the double array can be packed in a byte string
            # transform,
            strings,
            empty_buffer,
            v.simx_opmode_blocking)

        if res == v.simx_return_ok:
            cloud_handle = ret_ints[0]

            # convert the rgb values to a string
            color_size = 0
            if color_image is not None:
                # strings = [unicode(color_image.flatten().tostring(), 'utf-8')]
                strings = [display_name]
                empty_buffer = bytearray(color_image.tobytes())
                color_size = color_image.size
            else:
                strings = [display_name]

            # Actually transfer the point cloud
            res, ret_ints, ret_floats, ret_strings, ret_buffer = v.simxCallScriptFunction(
                self.client_id,
                'remoteApiCommandServer',
                v.sim_scripttype_childscript,
                'insertPointCloud_function',
                [parent_handle, transform_entries, points.size, cloud_handle, color_size],
                # np.append(transform, points),
                np.append(points, []),
                # for when the double array can be packed in a byte string
                # transform,
                strings,
                empty_buffer,
                v.simx_opmode_blocking)
            if res == v.simx_return_ok:
                print ('point cloud handle: ', ret_ints[0])  # display the reply from V-REP (in this case, the handle of the created dummy)
                self.setPose(display_name, transform, parent_handle)
                return ret_ints[0]
            else:
                print 'insertPointCloud_function remote function call failed.'
                return -1
        else:
            print('createPointCloud_function remote function call failed')

    def visualize(self, tf_session, dataset=FLAGS.grasp_dataset, batch_size=1, parent_name=FLAGS.vrepParentName):
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
            # load data from the next grasp attempt
            output_features_dict = tf_session.run(feature_op_dicts)
            if ((attempt_num >= FLAGS.vrepVisualizeGraspAttempt_min or FLAGS.vrepVisualizeGraspAttempt_min == -1) and
                    (attempt_num < FLAGS.vrepVisualizeGraspAttempt_max or FLAGS.vrepVisualizeGraspAttempt_max == -1)):
                for features_dict_np, sequence_dict_np in output_features_dict:
                    # TODO(ahundt) actually put transforms into V-REP or pybullet
                    self._visualize_one_grasp_attempt(
                            grasp_dataset_object, features_complete_list, features_dict_np, parent_handle,
                            dataset_name=dataset,
                            attempt_num=attempt_num)
            if (attempt_num > FLAGS.vrepVisualizeGraspAttempt_max and not FLAGS.vrepVisualizeGraspAttempt_max == -1):
                # stop running if we've gone through all the relevant attempts.
                break

    def _visualize_one_grasp_attempt(self, grasp_dataset_object, features_complete_list, features_dict_np, parent_handle,
                                     dataset_name=FLAGS.grasp_dataset,
                                     attempt_num=0,
                                     grasp_sequence_min_time_step=FLAGS.grasp_sequence_min_time_step,
                                     grasp_sequence_max_time_step=FLAGS.grasp_sequence_max_time_step,
                                     visualization_dir=FLAGS.visualization_dir,
                                     vrepDebugMode=FLAGS.vrepDebugMode,
                                     vrepVisualizeRGBD=FLAGS.vrepVisualizeRGBD,
                                     vrepVisualizeSurfaceRelativeTransform=FLAGS.vrepVisualizeSurfaceRelativeTransform):
        """Take an extracted grasp attempt tfrecord numpy dictionary and visualize it in vrep

        # Params

        parent_handle: the frame in which to display transforms, defaults to base frame of 'LBR_iiwa_14_R820'

        It is important to note that both V-REP and the grasp dataset use the xyzw quaternion format.
        """
        # get param strings for every single gripper position
        base_to_endeffector_transforms = grasp_dataset_object.get_time_ordered_features(
            features_complete_list,
            # feature_type='transforms/base_T_endeffector/vec_quat_7')  # display only commanded transforms
            # feature_type='vec_quat_7')  # display all transforms
            feature_type='reached_pose',
            step='move_to_grasp')
        print(features_complete_list)
        print(base_to_endeffector_transforms)
        camera_to_base_transform_name = 'camera/transforms/camera_T_base/matrix44'
        camera_intrinsics_name = 'camera/intrinsics/matrix33'

        pregrasp_depth_image_feature = grasp_dataset_object.get_time_ordered_features(
            features_complete_list,
            feature_type='depth_image/decoded',
            step='view_clear_scene'
        )[0]

        depth_image_features = grasp_dataset_object.get_time_ordered_features(
            features_complete_list,
            feature_type='depth_image/decoded',
            step='move_to_grasp'
        )

        rgb_image_features = grasp_dataset_object.get_time_ordered_features(
            features_complete_list,
            feature_type='/image/decoded',
            step='move_to_grasp'
        )

        grasp_success_feature_name = grasp_dataset_object.get_time_ordered_features(
            features_complete_list,
            feature_type='grasp_success'
        )[0]

        # Create repeated values for the final grasp position where the gripper closed
        base_T_endeffector_final_close_gripper_name = base_to_endeffector_transforms[-1]
        base_T_endeffector_final_close_gripper = features_dict_np[base_T_endeffector_final_close_gripper_name]

        # get the camera intrinsics matrix and camera extrinsics matrix
        camera_intrinsics_matrix = features_dict_np[camera_intrinsics_name]
        camera_to_base_4x4matrix = features_dict_np[camera_to_base_transform_name]
        print('camera/transforms/camera_T_base/matrix44: \n', camera_to_base_4x4matrix)
        camera_to_base_vec_quat_7 = grasp_geometry.matrix_to_vector_quaternion_array(camera_to_base_4x4matrix)
        camera_T_base_handle = self.create_dummy('camera_T_base', camera_to_base_vec_quat_7, parent_handle)
        # verify that another transform path gets the same result
        camera_T_base_ptrans = grasp_geometry.matrix_to_ptransform(camera_to_base_4x4matrix)
        camera_to_base_vec_quat_7_ptransform_conversion_test = grasp_geometry.ptransform_to_vector_quaternion_array(camera_T_base_ptrans)
        display_name = 'camera_T_base_vec_quat_7_ptransform_conversion_test'
        self.create_dummy(display_name, camera_to_base_vec_quat_7_ptransform_conversion_test, parent_handle)
        assert(grasp_geometry.vector_quaternion_arrays_allclose(camera_to_base_vec_quat_7, camera_to_base_vec_quat_7_ptransform_conversion_test))
        # verify that another transform path gets the same result
        base_T_camera_ptrans = camera_T_base_ptrans.inv()

        # TODO(ahundt) check that ptransform times its inverse is identity, or very close to it
        identity = sva.PTransformd.Identity()
        should_be_identity = base_T_camera_ptrans * camera_T_base_ptrans

        base_to_camera_vec_quat_7_ptransform_conversion_test = grasp_geometry.ptransform_to_vector_quaternion_array(base_T_camera_ptrans)
        display_name = 'base_to_camera_vec_quat_7_ptransform_conversion_test'
        self.create_dummy(display_name, base_to_camera_vec_quat_7_ptransform_conversion_test, parent_handle)
        # gripper_positions = [features_dict_np[base_T_endeffector_vec_quat_feature_name] for
        #                      base_T_endeffector_vec_quat_feature_name in base_to_endeffector_transforms]
        for i, base_T_endeffector_vec_quat_feature_name, depth_name, rgb_name in zip(range(len(base_to_endeffector_transforms)),
                                                                                     base_to_endeffector_transforms,
                                                                                     depth_image_features, rgb_image_features):
            # 2. Now create a dummy object at coordinate 0.1,0.2,0.3 with name 'MyDummyName':
            empty_buffer = bytearray()
            # 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
            base_T_endeffector_vec_quat_feature = features_dict_np[base_T_endeffector_vec_quat_feature_name]
            # display the raw base to endeffector feature
            bTe_display_name = str(i).zfill(2) + '_' + base_T_endeffector_vec_quat_feature_name.replace('/', '_')
            bTe_handle = self.create_dummy(bTe_display_name, base_T_endeffector_vec_quat_feature, parent_handle)

            # do the conversion needed for training
            camera_T_endeffector_ptrans, base_T_endeffector_ptrans, base_T_camera_ptrans = grasp_geometry.grasp_dataset_to_ptransform(
                camera_to_base_4x4matrix,
                base_T_endeffector_vec_quat_feature
            )
            # update the camera to base transform so we can visually ensure consistency
            camera_T_base_handle = self.create_dummy('camera_T_base', camera_to_base_vec_quat_7, parent_handle)
            base_to_camera_vec_quat_7 = grasp_geometry.ptransform_to_vector_quaternion_array(base_T_camera_ptrans)
            base_T_camera_handle = self.create_dummy('base_T_camera', base_to_camera_vec_quat_7, parent_handle)

            # test that the base_T_endeffector -> ptransform -> vec_quat_7 roundtrip returns the same transform
            base_T_endeffector_vec_quat = grasp_geometry.ptransform_to_vector_quaternion_array(base_T_endeffector_ptrans)
            bTe_display_name = str(i).zfill(2) + '_base_T_endeffector_ptransform_conversion_test_' + base_T_endeffector_vec_quat_feature_name.replace('/', '_')
            self.create_dummy(bTe_display_name, base_T_endeffector_vec_quat, parent_handle)
            assert(grasp_geometry.vector_quaternion_arrays_allclose(base_T_endeffector_vec_quat_feature, base_T_endeffector_vec_quat))

            # verify that another transform path gets the same result
            # camera_to_base_vec_quat_7_ptransform_conversion_test = grasp_geometry.ptransform_to_vector_quaternion_array(camera_T_base_ptrans)
            # display_name = str(i).zfill(2) + '_camera_to_base_vec_quat_7_ptransform_conversion_test'
            # self.create_dummy(display_name, camera_to_base_vec_quat_7_ptransform_conversion_test, parent_handle)

            cTe_display_name = str(i).zfill(2) + '_camera_T_endeffector_' + base_T_endeffector_vec_quat_feature_name.replace(
                '/transforms/base_T_endeffector/vec_quat_7', '').replace('/', '_')
            cTe_vec_quat = grasp_geometry.ptransform_to_vector_quaternion_array(camera_T_endeffector_ptrans)
            self.create_dummy(cTe_display_name, cTe_vec_quat, base_T_camera_handle)

            if i == 0:
                clear_frame_depth_image = np.squeeze(features_dict_np[pregrasp_depth_image_feature])
            # format the dummy string nicely for display
            transform_display_name = str(i).zfill(2) + '_' + base_T_endeffector_vec_quat_feature_name.replace(
                '/transforms/base_T_endeffector/vec_quat_7', '').replace('/', '_')
            print base_T_endeffector_vec_quat_feature_name, transform_display_name, base_T_endeffector_vec_quat_feature
            # display the gripper pose
            self.create_dummy(transform_display_name, base_T_endeffector_vec_quat_feature, parent_handle)
            # Perform some consistency checks based on the above
            assert(grasp_geometry.vector_quaternion_arrays_allclose(base_T_endeffector_vec_quat, base_T_endeffector_vec_quat_feature))

            #############################
            # get the transform from the current endeffector pose to the final, plus include the angle theta in the name
            transform_display_name = str(i).zfill(2) + '_current_T_end'
            current_to_end = grasp_geometry.current_endeffector_to_final_endeffector_feature(
                base_T_endeffector_vec_quat_feature, base_T_endeffector_final_close_gripper, feature_type='vec_quat_7')
            current_to_end_ptransform = grasp_geometry.vector_quaternion_array_to_ptransform(current_to_end)
            current_to_end_rotation = current_to_end_ptransform.rotation()
            theta = grasp_geometry.grasp_dataset_rotation_to_theta(current_to_end_rotation, verbose=1)
            # compare these printed theta values in the visualization to what is documented in
            # see grasp_dataset_rotation_to_theta() this printout will let you verify that
            # theta is estimated correctly for training.
            print('current to end estimated theta ', transform_display_name, theta)
            self.create_dummy(transform_display_name, current_to_end, bTe_handle)

            # TODO(ahundt) check that transform from end step to itself should be identity, or very close to it
            # if base_T_endeffector_final_close_gripper_name == base_T_endeffector_vec_quat_feature_name:
            #     transform from end step to itself should be identity.
            #     identity = sva.PTransformd.Identity()
            #     assert(identity == current_to_end)

            #############################
            # visualize surface relative transform
            if vrepVisualizeSurfaceRelativeTransform:
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
                surface_relative_transform_display_name = str(i).zfill(2) + '_depth_point_T_endeffector'
                surface_relative_transform_dummy_handle = self.create_dummy(surface_relative_transform_display_name,
                                                                            surface_relative_transform_vec_quat,
                                                                            depth_point_dummy_handle)

            if(vrepVisualizeRGBD and i >= FLAGS.vrepVisualizeRGBD_min and i < FLAGS.vrepVisualizeRGBD_max):
                self.visualize_rgbd(features_dict_np, rgb_name, depth_name, grasp_sequence_min_time_step, i, grasp_sequence_max_time_step,
                                    camera_intrinsics_matrix, vrepDebugMode, dataset_name, attempt_num, grasp_success_feature_name,
                                    visualization_dir, base_to_camera_vec_quat_7, parent_handle)
        print('visualization of grasp attempt #', attempt_num, ' complete')

    def visualize_rgbd(self, features_dict_np, rgb_name, depth_name, grasp_sequence_min_time_step, i,
                       grasp_sequence_max_time_step, camera_intrinsics_matrix, vrepDebugMode, dataset_name, attempt_num,
                       grasp_success_feature_name, visualization_dir, camera_to_base_vec_quat_7, parent_handle):
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
            point_cloud_detailed_name = ('point_cloud_' + str(dataset_name) + '_' + str(attempt_num) + '_' + str(i).zfill(2) +
                                         '_rgbd_' + depth_name.replace('/depth_image/decoded', '').replace('/', '_') +
                                         '_success_' + str(int(features_dict_np[grasp_success_feature_name])))
            print(point_cloud_detailed_name)

            # print 'point_cloud:', point_cloud.transpose()[:30, :3]
            path = os.path.join(visualization_dir, point_cloud_detailed_name + '.ply')
            print 'point_cloud.size:', point_cloud.size
            xyz = point_cloud.reshape([point_cloud.size/3, 3])
            rgb = np.squeeze(rgb_image).reshape([point_cloud.size/3, 3])

            # Save out Point cloud
            if 'save_ply' in vrepDebugMode:
                write_xyz_rgb_as_ply(xyz, rgb, path)
            point_cloud_display_name = str(i).zfill(2) + '_point_cloud'
            self.create_point_cloud(point_cloud_display_name, xyz, camera_to_base_vec_quat_7, rgb, parent_handle)

            # display the rgb and depth image
            self.display_images(rgb, depth_image_float_format)

    def display_images(self, rgb, depth_image_float_format):
        """Display the rgb and depth image in V-REP (not yet working)

        Reference code: https://github.com/nemilya/vrep-api-python-opencv/blob/master/handle_vision_sensor.py
        V-REP Docs: http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm#simxSetVisionSensorImage
        """
        res, kcam_rgb_handle = v.simxGetObjectHandle(self.client_id, 'kcam_rgb', v.simx_opmode_oneshot_wait)
        print('kcam_rgb_handle: ', kcam_rgb_handle)
        rgb_for_display = rgb.astype('uint8')
        rgb_for_display = rgb_for_display.ravel()
        is_color = 1
        res = v.simxSetVisionSensorImage(self.client_id, kcam_rgb_handle, rgb_for_display, is_color, v.simx_opmode_oneshot)
        print('simxSetVisionSensorImage rgb result: ', res, ' rgb shape: ', rgb.shape)
        res, kcam_depth_handle = v.simxGetObjectHandle(self.client_id, 'kcam_depth', v.simx_opmode_oneshot_wait)
        normalized_depth = depth_image_float_format * 255 / depth_image_float_format.max()
        normalized_depth = normalized_depth.astype('uint8')
        normalized_depth = normalized_depth.ravel()
        is_color = 0
        res = v.simxSetVisionSensorImage(self.client_id, kcam_depth_handle, normalized_depth, is_color, v.simx_opmode_oneshot)
        print('simxSetVisionSensorImage depth result: ', res, ' depth shape: ', depth_image_float_format.shape)

    def __del__(self):
        v.simxFinish(-1)


if __name__ == '__main__':

    with tf.Session() as sess:
        sim = VREPGraspSimulation()
        sim.visualize(sess)

