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
    qa4 = eigen.Vector4d(vector_quaternion_array[3:])
    q = eigen.Quaterniond(qa4)
    # The ptransform needs the rotation component inverted.
    # see https://github.com/ahundt/grl/blob/master/include/grl/vrep/SpaceVecAlg.hpp#L22
    q = q.inverse()
    pt = sva.PTransformd(q,v)
    return pt


def ptransform_to_vector_quaternion_array(ptransform):
    """Convert a PTransformD into a vector quaternion array
    containing 3 vector entries (x, y, z) and 4 quaternion entries (x, y, z, w)
    """
    rot = ptransform.rotation()
    quaternion = eigen.Quaterniond(rot)
    quaternion = quaternion.inverse()
    translation = ptransform.translation()
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
    """Convert a 4x4 homogeneous transformation matrix into a ptransform
    """
    vq = matrix_to_vector_quaternion_array(matrix)
    pt = vector_quaternion_array_to_ptransform(vq)
    return pt


def vector_to_ptransform(XYZ):
    """Convert a 3 element translation vector into a ptransform with no rotation
    """
    q = eigen.Quaterniond()
    q = q.setIdentity()
    v = eigen.Vector3d(XYZ)
    ptransform = sva.PTransformd(q, v)
    return ptransform


def depth_image_pixel_to_cloud_point(depth_image, camera_intrinsics_matrix, pixel_coordinate, augmentation_rectangle=None):
    """Convert a single specific depth image pixel coordinate into a point cloud point.

    # Params

    depth_image:
        width x height x depth image in floating point format
    camera_intrinsics_matrix:
        'camera/intrinsics/matrix33' The 3x3 camera intrinsics matrix
        used to convert depth to point cloud points.
    pixel_coordinate:
        The x, y depth image pixel coordinate of the depth image pixel to convert.
    augmentation_rectangle:
       A random offset for the selected (dx, dy) pixel index.
       It will randomly select a pixel in a box around the endeffector coordinate.
       Default (1, 1) has no augmentation.
    """
    # The frame definitions switch up a bit here, the calculation of the
    # gripper pose in the image frame is done with the graphics coordinate
    # convention where:
    # - Y is depth
    # - X is right in the image frame
    # - Z is up in the image frame

    # get the point index in the depth image
    # TODO(ahundt) is this the correct indexing scheme, are any axes flipped?
    x = pixel_coordinate[0]
    z = pixel_coordinate[1]

    # choose a random pixel in the specified box
    if(augmentation_rectangle is not None and
       augmentation_rectangle is not (1, 1)):
        # Add a random coordinate offset for the depth data
        # to augment the surface relative transforms
        x_max = np.ceil((augmentation_rectangle[0]-1)/2)
        y_max = np.ceil((augmentation_rectangle[1]-1)/2)
        x += np.randint(-x_max, x_max)
        z += np.randint(-y_max, y_max)

    # get focal length and camera image center from the intrinsics matrix
    fx = camera_intrinsics_matrix[0, 0]
    fy = camera_intrinsics_matrix[1, 1]
    center_x = camera_intrinsics_matrix[0, 2]
    center_y = camera_intrinsics_matrix[1, 2]

    # Capital Y is depth in camera frame
    Y = depth_image[x, z]
    # Capital X is horizontal point, right in camera image frame
    X = (x - center_x) * Y / fx
    # Capital Z is vertical point, up in camera image frame
    Z = (z - center_y) * Y / fy
    return np.array((X, Y, Z))


def surface_relative_transform(depth_image,
                               camera_intrinsics_matrix,
                               camera_T_endeffector,
                               augmentation_rectangle=None,
                               return_depth_image_coordinate=False):
    """Get the transform from a depth pixel to a gripper pose with optional data augmentation.

    # Params

    depth_image:
        width x height x depth image in floating point format, depth in meters
    camera_intrinsics_matrix:
        'camera/intrinsics/matrix33' The 3x3 camera intrinsics matrix.
    camera_T_endeffector:
       PTransformd that takes a point in the endeffector frame and transforms
       it to a point in the camera frame.
    augmentation_rectangle:
       A random offset for the selected (dx, dy) pixel index.
       It will randomly select a pixel in a box around the endeffector coordinate.
       Default (1, 1) has no augmentation.
    return_depth_image_coordinate:
       changes the return to include the x, y coordinate used for the depth image.

    # Returns

       [x, y, z, qx, qy, qz, qw] when return_depth_image_coordinate is False.
       When return_depth_image_coordinate is True:
           Numpy array [x, y, z, qx, qy, qz, qw, dx, dy], which contains:
           - vector (x, y, z) for cartesian motion
           - quaternion (qx, qy, qz, qw) for rotation
           - The selected (dx, dy) pixel width, height coordinate in the depth image.
             This coordinate is used to calculate the point cloud point used for the
             surface relative transform.
    """
    # xyz coordinate of the endeffector in the camera frame
    XYZ, pixel_coordinate_of_endeffector = endeffector_image_coordinate_and_cloud_point(
        depth_image, camera_intrinsics_matrix, camera_T_endeffector, augmentation_rectangle,)

    # make an identity quaternion because the pixel will use the camera orientation
    # TODO(ahundt) is this the right axis ordering for the translation component
    camera_T_cloud_point_ptrans = vector_to_ptransform(XYZ)
    # TODO(ahundt) is this inverse correct?
    cloud_point_T_camera_ptrans = camera_T_cloud_point_ptrans.inverse()
    # transform point all the way to depth frame
    depth_pixel_T_endeffector_ptrans = cloud_point_T_camera_ptrans * camera_T_endeffector
    # get the depth relative transform
    # TODO(ahundt) maybe the rotation component of this needs to be inverted due to sva implementation?
    depth_relative_vec_quat_array = ptransform_to_vector_quaternion_array(depth_pixel_T_endeffector_ptrans)

    if return_depth_image_coordinate:
        # return the transform and the image coordinate used to generate the transform
        image_x = pixel_coordinate_of_endeffector[0]
        image_y = pixel_coordinate_of_endeffector[1]
        return np.concatenate((depth_relative_vec_quat_array, image_x, image_y))
    else:
        return depth_relative_vec_quat_array


def endeffector_image_coordinate(camera_intrinsics_matrix, xyz):

    # get focal length and camera image center from the intrinsics matrix
    fx = camera_intrinsics_matrix[0, 0]
    fy = camera_intrinsics_matrix[1, 1]
    center_x = camera_intrinsics_matrix[2, 0]
    center_y = camera_intrinsics_matrix[2, 1]

    # Capital X is horizontal point, right in camera image frame
    X = xyz[0]
    # Capital Y is depth in camera frame
    Y = xyz[1]
    # Capital Z is vertical point, up in camera image frame
    Z = xyz[2]
    # x is the image coordinate horizontal axis
    x = (X * fx / Y) + center_x
    # y is the image coordinate vertical axis
    y = (Z * fy / Y) + center_y
    return np.array([x, y])


def endeffector_image_coordinate_and_cloud_point(depth_image,
                                                 camera_intrinsics_matrix,
                                                 camera_T_endeffector,
                                                 augmentation_rectangle=None):
    """Get the xyz coordinate of the endeffector in the camera frame as well as its image coordinate.

    # Returns

    [XYZ, pixel_coordinate_of_endeffector]

    xyz: the xyz coordinate of the end effector's point cloud point.
    pixel_coordinate_of_endeffector: the x, y coordinate in the depth image frame of the xyz point cloud point.
    """
    # xyz coordinate of the endeffector in the camera frame
    cte_xyz = camera_T_endeffector.translation()
    # transform the end effector coordinate into the depth image coordinate
    pixel_coordinate_of_endeffector = endeffector_image_coordinate(camera_intrinsics_matrix, cte_xyz)

    # The frame definitions switch up a bit here, the calculation of the
    # gripper pose in the image frame is done with the graphics coordinate
    # convention where:
    # - Y is depth
    # - X is right in the image frame
    # - Z is up in the image frame
    XYZ = depth_image_pixel_to_cloud_point(depth_image,
                                           camera_intrinsics_matrix,
                                           pixel_coordinate_of_endeffector,
                                           augmentation_rectangle=augmentation_rectangle)
    return XYZ, pixel_coordinate_of_endeffector


def brainrobotdata_to_ptransform(camera_T_base, base_T_endeffector):
    """Convert brainrobotdata features camera_T_base and base_T_endeffector to base_T_endeffector and ptransforms.

    This specific function exists because it accepts the raw feature types
    defined in the google brain robot grasping dataset.

    # Params

    camera_T_base: a vector quaternion array
    base_T_endeffector: a 4x4 homogeneous 3D transformation matrix

    # Returns

      PTransformd formatted transforms:
      camera_T_endeffector_ptrans, base_T_endeffector_ptrans, camera_T_base_ptrans
    """
    base_T_endeffector_ptrans = vector_quaternion_array_to_ptransform(base_T_endeffector)
    # In this case camera_T_base is a transform that takes a point in the base
    # frame of reference and transforms it to the camera frame of reference.
    camera_T_base_ptrans = matrix_to_ptransform(camera_T_base)
    camera_T_endeffector_ptrans = camera_T_base_ptrans * base_T_endeffector_ptrans
    return camera_T_endeffector_ptrans, base_T_endeffector_ptrans, camera_T_base_ptrans


def brainrobotdata_to_surface_relative_transform(depth_image,
                                                 camera_intrinsics_matrix,
                                                 camera_T_base,
                                                 base_T_endeffector,
                                                 augmentation_rectangle=None,
                                                 return_depth_image_coordinate=False):
    """Get the transform from a depth pixel to a gripper pose from data in the brain robot data feature formats.

    This specific function exists because it accepts the raw feature types
    defined in the google brain robot grasping dataset.

    Includes optional data augmentation.

    # Params

    depth_image:
        width x height x depth image in floating point format
    camera_intrinsics_matrix:
        'camera/intrinsics/matrix33' The 3x3 camera intrinsics matrix.
    camera_T_base:
        'camera/transforms/camera_T_base/matrix44'
        4x4 transformation matrix from the camera center to the robot base.
        camera_T_base is a transform that takes a point in the base
        frame of reference and transforms it to the camera frame of reference.
    base_T_endeffector:
       vector (x, y, z) for cartesian motion and quaternion (qx, qy, qz, qw) for rotation.
       base_T_endeffector is a transform that takes a point in the endeffector
       frame of reference and transforms it to the base frame of reference.
    augmentation_rectangle:
       A random offset for the selected (dx, dy) pixel index.
       It will randomly select a pixel in a box around the endeffector coordinate.
       Default (1, 1) has no augmentation.

    return_depth_image_coordinate:
       changes the return

    # Returns

       [x, y, z, qx, qy, qz, qw] when return_depth_image_coordinate is False.
       When return_depth_image_coordinate is True:
           Numpy array [x, y, z, qx, qy, qz, qw, dx, dy], which contains:
           - vector (x, y, z) for cartesian motion
           - quaternion (qx, qy, qz, qw) for rotation
           - The selected (dx, dy) pixel width, height coordinate in the depth image.
             This coordinate is used to calculate the point cloud point used for the
             surface relative transform.
    """
    camera_T_endeffector_ptrans, _, _ = brainrobotdata_to_ptransform(camera_T_base, base_T_endeffector)
    return surface_relative_transform(depth_image,
                                      camera_intrinsics_matrix,
                                      camera_T_endeffector_ptrans,
                                      augmentation_rectangle,
                                      return_depth_image_coordinate)


def current_endeffector_to_final_endeffector_ptransform(current_base_T_endeffector, end_base_T_endeffector):
    """Calculate the ptransform between two poses in the same base frame.

       A pose is a 6 degree of freedom rigid transform represented with 7 values:
       vector (x, y, z) and quaternion (x, y, z, w).
       A pose is always annotated with the target and source frames of reference.
       For example, base_T_camera is a transform that takes a point in the camera
       frame of reference and transforms it to the base frame of reference.

       We will be dealing with, for example:
       grasp/4/reached_pose/transforms/base_T_endeffector/vec_quat_7
       grasp/10/commanded_pose/transforms/base_T_endeffector/vec_quat_7

       # Params

       current_base_T_endeffector: A vector quaternion array from a base frame to an end effector frame
       end_base_T_endeffector: A vector quaternion array from a base frame to an end effector frame
    """
    base_to_current = vector_quaternion_array_to_ptransform(current_base_T_endeffector)
    base_to_end = vector_quaternion_array_to_ptransform(end_base_T_endeffector)
    inv_b2c = base_to_current.inverse()
    current_to_end = inv_b2c * base_to_end
    # we have ptransforms for both data, now get transform from current to commanded
    current_to_end = ptransform_to_vector_quaternion_array(current_to_end)
    return current_to_end
