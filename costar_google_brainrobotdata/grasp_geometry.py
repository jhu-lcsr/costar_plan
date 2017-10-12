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


def tfrecordPoseToPtransform(np):
    """A pose is a 6 degree of freedom rigid transform represented with 7 values:
       vector (x, y, z) and quaternion (x, y, z, w).
       eigen Quaterniond is also ordered xyzw
    """
    v = eigen.Vector3d(np[:3])
    qa4 = eigen.Vector4d(np[4:])
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
    base_to_current = tfrecordPoseToPtransform(currentPoseReached)
    base_to_end = tfrecordPoseToPtransform(endPoseCommanded)
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


def matrix_to_vector_quaternion_array(matrix, inverse=False):
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
    print vec_quat_7
    return vec_quat_7