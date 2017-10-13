# 3D geometry algorithms for calculating deep learning grasp algorithm input parameters.
#
# Copyright 2017 Andrew Hundt 2017.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
try:
    import eigen  # https://github.com/jrl-umi3218/Eigen3ToPython
    import sva  # https://github.com/jrl-umi3218/SpaceVecAlg
except ImportError:
    print('eigen and sva not available, skipping components utilizing 3D geometry algorithms.'
          'To install run the script at'
          'https://github.com/ahundt/robotics_setup/blob/master/robotics_tasks.sh'
          'or follow the instructions at https://github.com/jrl-umi3218/SpaceVecAlg'
          'and https://github.com/jrl-umi3218/SpaceVecAlg and make sure python bindings'
          'are enabled.')


def vector_quaternion_array_to_ptransform(vector_quaternion_array):
    """Convert a vector and quaternion pose array to a plucker transform.

    # Params

        vector_quaternion_array: A numpy array containing a pose.
        A pose is a 6 degree of freedom rigid transform represented with 7 values:
        [x, y, z, qx, qy, qz, qw] consisting of a
        vector (x, y, z) for cartesian motion and quaternion (qx, qy, qz, qw) for rotation.
        This is the data format used by the google brain robot data grasping dataset's
        tfrecord poses.
        eigen Quaterniond is also ordered xyzw.

    # Returns

      A plucker transform as defined by the spatial vector algebra library.
      https://github.com/jrl-umi3218/SpaceVecAlg
      https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
    """
    v = eigen.Vector3d(vector_quaternion_array[:3])
    qa4 = eigen.Vector4d(vector_quaternion_array[4:])
    q = eigen.Quaterniond(qa4)
    # The ptransform needs the rotation component inverted.
    # see https://github.com/ahundt/grl/blob/master/include/grl/vrep/SpaceVecAlg.hpp#L22
    q = q.inverse()
    pt = sva.PTransformd(q,v)
    return pt


def currentPoseToEndPose(currentPoseReached, endPoseCommanded):
    """A pose is a 6 degree of freedom rigid transform represented with 7 values:
       vector (x, y, z) and quaternion (x, y, z, w).
       A pose is always annotated with the target and source frames of reference.
       For example, base_T_camera is a transform that takes a point in the camera
       frame of reference and transforms it to the base frame of reference.

       We will be dealing with, for example:
       grasp/4/reached_pose/transforms/base_T_endeffector/vec_quat_7
       grasp/10/commanded_pose/transforms/base_T_endeffector/vec_quat_7
    """
    base_to_current = vector_quaternion_array_to_ptransform(currentPoseReached)
    base_to_end = vector_quaternion_array_to_ptransform(endPoseCommanded)
    inv_b2c = base_to_current.inverse()
    current_to_end = inv_b2c * base_to_end
    # we have ptransforms for both data, now get transform from current to commanded
    current_to_end = ptransform_to_vector_quaternion_array(current_to_end)
    return current_to_end


def ptransform_to_vector_quaternion_array(pt):
    """Convert a PTransformD into a vector quaternion array
    containing 3 vector entries (x, y, z) and 4 quaternion entries (x, y, z, w)
    """
    rot = pt.rotation()
    quaternion = eigen.Quaterniond(rot)
    translation = current_to_end.translation()
    translation = np.array(translation).reshape(3)
    q_floats_array = np.array(quaternion.coeffs()).astype(np.float32)
    vec_quat_7 = np.append(translation, q_floats_array)
    return vec_quat_7


def matrix_to_vector_quaternion_array(matrix, inverse=False, verbose=0):
    """Convert a 4x4 Rt transformation matrix into a vector quaternion array
    containing 3 vector entries (x, y, z) and 4 quaternion entries (x, y, z, w)
    """
    rot = eigen.Matrix3d(matrix[:3, :3])
    quaternion = eigen.Quaterniond(rot)
    translation = matrix[:3, 3].transpose()
    if inverse:
        quaternion = quaternion.inverse()
        translation *= -1
    q_floats_array = np.array(quaternion.coeffs()).astype(np.float32)
    vec_quat_7 = np.append(translation, q_floats_array)
    if verbose > 0:
        print(vec_quat_7)
    return vec_quat_7


def matrix_to_ptransform(matrix):
    vq = matrix_to_vector_quaternion_array(matrix)
    pt = vector_quaternion_array_to_ptransform(vq)
    return pt


def surface_relative_transform(depth_image,
                               camera_intrinsics_matrix,
                               camera_T_base,
                               base_T_endeffector):
    """Get the transform from a depth pixel to a gripper pose.

    # Params

    camera_intrinsics:
        'camera/intrinsics/matrix33' The 3x3 camera intrinsics matrix.
    camera_T_base:
        'camera/transforms/camera_T_base/matrix44'
        Same as base_T_endeffector but from the camera center to the robot base,
        and contains a 4x4 transformation matrix instead of a vector and quaternion.
    base_T_endeffector
       vector (x, y, z) for cartesian motion and quaternion (qx, qy, qz, qw) for rotation.
    """
    # In this case base_T_endeffector is a transform that takes a point in the endeffector
    # frame of reference and transforms it to the base frame of reference.
    base_T_endeffector_pt = vector_quaternion_array_to_ptransform(base_T_endeffector)
    # In this case camera_T_base is a transform that takes a point in the base
    # frame of reference and transforms it to the camera frame of reference.
    camera_T_base_pt = matrix_to_ptransform(camera_T_base)
    # In this case camera_T_base is a transform that takes a point in the base
    # frame of reference and transforms it to the camera frame of reference.
    camera_T_endeffector_pt = camera_T_base_pt * base_T_endeffector_pt

    # TODO(ahundt) maybe this needs to be inverted?
    # xyz coordinate of the endeffector in the camera frame
    cte_xyz = camera_T_endeffector_pt.translation()
    # transform the end effector coordinate into the depth image coordinate
    # TODO(ahundt) is this multiplication ordering correct?
    pixel_coordinate_of_endeffector = camera_intrinsics_matrix * cte_xyz

    # get the point index in the depth image
    x = pixel_coordinate_of_endeffector[0]
    z = pixel_coordinate_of_endeffector[1]

    # get focal length and camera image center from the intrinsics matrix
    fx = camera_intrinsics_matrix[0, 0]
    fy = camera_intrinsics_matrix[1, 1]
    center_x = camera_intrinsics_matrix[0, 2]
    center_y = camera_intrinsics_matrix[1, 2]

    # Capital Y is depth in camera frame
    Y = depth_image[x, z]
    # Capital X is horizontal point, right in camera image frame
    X = (x - center_x) * Y / fx
    # Capital Z is verical point, up in camera image frame
    Z = (z - center_y) * Y / fy

    q = eigen.Quaterniond()
    q = q.setIdentity()
    v = eigen.Vector3d(np.array([X, Y, Z]))
    # multiply back out to the point cloud location
    camera_T_cloud_point = sva.PTransformd(q,v)
    # TODO(ahundt) is this inverse correct?
    cloud_point_T_camera = camera_T_cloud_point.inverse()
    # transform point all the way to depth frame
    # TODO(ahundt) is this multiplication ordering correct?
    depth_pixel_T_endeffector = cloud_point_T_camera * camera_T_endeffector_pt
    # get the depth value
    # return new vector and quaternion
    depth_relative_vec_quat_array = ptransform_to_vector_quaternion_array(depth_pixel_T_endeffector)
    return depth_relative_vec_quat_array


