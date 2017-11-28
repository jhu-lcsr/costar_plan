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
          'or follow the instructions at https://github.com/jrl-umi3218/Eigen3ToPython'
          'and https://github.com/jrl-umi3218/SpaceVecAlg and make sure python bindings'
          'are enabled.')


def vector_quaternion_array_to_ptransform(vector_quaternion_array, q_inverse=True, t_inverse=False, pt_inverse=False):
    """Convert a vector and quaternion pose array to a plucker transform.

    # Params

        vector_quaternion_array: A numpy array containing a pose.
        A pose is a 6 degree of freedom rigid transform represented with 7 values:
        [x, y, z, qx, qy, qz, qw] consisting of a
        vector (x, y, z) for cartesian motion and quaternion (qx, qy, qz, qw) for rotation.
        This is the data format used by the google brain robot data grasping dataset's
        tfrecord poses.
        eigen Quaterniond is also ordered xyzw.
        q_inverse: Invert the quaternion before it is loaded into the PTransformd, defaults to True.
            With a PTransform the rotation component must be inverted before loading and after extraction.
            See https://github.com/ahundt/grl/blob/master/include/grl/vrep/SpaceVecAlg.hpp#L22 for a well tested example.
            See https://github.com/jrl-umi3218/Tasks/issues/10 for a detailed discussion leading to this conclusion.
            The default for q_inverse should be True, do not switch the q_inverse default to False
            without careful consideration and testing, though such a change may be appropriate when
            loading into another transform representation in which the rotation component is expected
            to be inverted.

    # Returns

      A plucker transform PTransformd object as defined by the spatial vector algebra library.
      https://github.com/jrl-umi3218/SpaceVecAlg
      https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
    """
    v = eigen.Vector3d(vector_quaternion_array[:3])
    if t_inverse is True:
        v *= -1
    # TODO(ahundt) use following lines after https://github.com/jrl-umi3218/Eigen3ToPython/pull/15 is fixed
    # qa4 = eigen.Vector4d()
    # q = eigen.Quaterniond(qa4)

    # Quaterniond important coefficient ordering details:
    # scalar constructor is Quaterniond(w,x,y,z)
    # vector constructor is Quaterniond(np.array([x,y,z,w]))
    # Quaterniond.coeffs() is [x,y,z,w]
    # https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html
    xyzw = eigen.Vector4d(vector_quaternion_array[3:])
    q = eigen.Quaterniond(xyzw)

    # The ptransform needs the rotation component to inverted before construction.
    # see https://github.com/ahundt/grl/blob/master/include/grl/vrep/SpaceVecAlg.hpp#L22 for a well tested example
    # see https://github.com/jrl-umi3218/Tasks/issues/10 for a detailed discussion leading to this conclusion
    if q_inverse is True:
        q = q.inverse()
    pt = sva.PTransformd(q,v)
    if pt_inverse is True:
        pt = pt.inv()
    return pt


def ptransform_to_vector_quaternion_array(ptransform, q_inverse=True, dtype=np.float32):
    """Convert a PTransformD into a vector quaternion array
    containing 3 vector entries (x, y, z) and 4 quaternion entries (x, y, z, w)

    # Params

    ptransform: The plucker transform from sva.PTransformd to be converted
    q_inverse: Invert the quaternion after it is extracted, defaults to True.
    With a PTransform the rotation component must be inverted before loading and after extraction.
    See https://github.com/ahundt/grl/blob/master/include/grl/vrep/SpaceVecAlg.hpp#L22 for a well tested example.
    See https://github.com/jrl-umi3218/Tasks/issues/10 for a detailed discussion leading to this conclusion.
    The default for q_inverse should be True, do not switch the q_inverse default to False
    without careful consideration and testing, though such a change may be appropriate when
    loading into another transform representation in which the rotation component is expected
    to be inverted.
    TODO(ahundt) in process of debugging, correct docstring when all issues are resolved.
    """
    rot = ptransform.rotation()
    quaternion = eigen.Quaterniond(rot)
    if q_inverse:
        quaternion = quaternion.inverse()
    translation = ptransform.translation()
    translation = np.array(translation).reshape(3)
    # coeffs are in xyzw order
    q_floats_array = np.array(quaternion.coeffs())
    vec_quat_7 = np.append(translation, q_floats_array)
    return vec_quat_7.astype(dtype)


def matrix_to_vector_quaternion_array(matrix, inverse=False, verbose=0):
    """Convert a 4x4 Rt transformation matrix into a vector quaternion array
    containing 3 vector entries (x, y, z) and 4 quaternion entries (x, y, z, w)

    # Params

        matrix: The 4x4 Rt rigid body transformation matrix to convert into a vector quaternion array.
        inverse: Inverts the full matrix before loading into the array.
            Useful when the transformation to be reversed and for testing/debugging purposes.

    # Returns

      np.array([x, y, z, qx, qy, qz, qw])
    """
    rot = eigen.Matrix3d(matrix[:3, :3])
    quaternion = eigen.Quaterniond(rot)
    translation = matrix[:3, 3].transpose()
    if inverse:
        quaternion = quaternion.inverse()
        translation *= -1
    # coeffs are in xyzw order
    q_floats_array = np.array(quaternion.coeffs())
    # q_floats_array = np.array([quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()]).astype(np.float32)
    vec_quat_7 = np.append(translation, q_floats_array)
    if verbose > 0:
        print(vec_quat_7)
    return vec_quat_7


def matrix_to_ptransform(matrix, q_inverse=True, t_inverse=False, pt_inverse=False):
    """Convert a 4x4 homogeneous transformation matrix into an sva.PTransformd plucker ptransform object.
    """
    vq = matrix_to_vector_quaternion_array(matrix)
    pt = vector_quaternion_array_to_ptransform(vq, q_inverse=q_inverse, t_inverse=t_inverse, pt_inverse=pt_inverse)
    return pt


def vector_to_ptransform(XYZ):
    """Convert (x,y,z) translation to sva.Ptransformd.

    Convert an xyz 3 element translation vector to an sva.PTransformd plucker
    ptransform object with no rotation applied. In other words,
    the rotation component will be the identity rotation.
    """
    q = eigen.Quaterniond()
    q.setIdentity()
    v = eigen.Vector3d(XYZ)
    ptransform = sva.PTransformd(q, v)
    return ptransform


def depth_image_pixel_to_cloud_point(depth_image,
                                     camera_intrinsics_matrix,
                                     pixel_coordinate,
                                     augmentation_rectangle=None,
                                     flip_x=1.0,
                                     flip_y=1.0):
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
    # get the point index in the depth image
    # TODO(ahundt) is this the correct indexing scheme, are any axes flipped?
    x = int(pixel_coordinate[0])
    y = int(pixel_coordinate[1])

    # choose a random pixel in the specified box
    if(augmentation_rectangle is not None and
       augmentation_rectangle is not (1, 1)):
        # Add a random coordinate offset for the depth data
        # to augment the surface relative transforms
        x_max = np.ceil((augmentation_rectangle[0]-1)/2)
        y_max = np.ceil((augmentation_rectangle[1]-1)/2)
        x += np.randint(-x_max, x_max)
        y += np.randint(-y_max, y_max)

    # get focal length and camera image center from the intrinsics matrix
    fx = camera_intrinsics_matrix[0, 0]
    fy = camera_intrinsics_matrix[1, 1]
    center_x = camera_intrinsics_matrix[2, 0]
    center_y = camera_intrinsics_matrix[2, 1]

    if x < 0 or x >= depth_image.shape[0] or y < 0 or y >= depth_image.shape[1]:
        print('warning: attempting to access pixel outside of image dimensions, '
              'choosing center pixel instead.')
        x = depth_image.shape[0]/2
        y = depth_image.shape[1]/2

    # Capital Z is depth in camera frame
    Z = depth_image[x, y]
    # Capital X is horizontal point, right in camera image frame
    X = flip_x * (x - center_x) * Z / fx
    # Capital Y is vertical point, up in camera image frame
    Y = flip_y * (y - center_y) * Z / fy
    XYZ = np.array([X, Y, Z])
    return XYZ


def surface_relative_transform(depth_image,
                               camera_intrinsics_matrix,
                               camera_T_endeffector,
                               augmentation_rectangle=None):
    """Get (1) the transform from a depth pixel to a gripper pose, and (2) the corresponding image coordinate.

    Includes optional data augmentation (Not yet implemented).

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

    # Returns

        [depth_pixel_T_endeffector_ptrans, image_coordinate]

        depth_pixel_T_endeffector_ptrans is an sva.PTransformd pose from a depth pixel coordinate
        assigned the same orientation as the camera to the position and orientation of the gripper.

        The image_coordinate (dx, dy) is the pixel width, height
        coordinate of the transform in the depth image.

        image_coordinate is used to calculate the point cloud point used for the
        surface relative transform and to generate 2D label weights.
    """
    # xyz coordinate of the endeffector in the camera frame
    XYZ, pixel_coordinate_of_endeffector = endeffector_image_coordinate_and_cloud_point(
        depth_image, camera_intrinsics_matrix, camera_T_endeffector, augmentation_rectangle)

    # Convert the point cloud point to a transform with identity rotation
    camera_T_cloud_point_ptrans = vector_to_ptransform(XYZ)
    # get the depth pixel to endeffector transform, aka surface relative transform
    depth_pixel_T_endeffector_ptrans = camera_T_endeffector * camera_T_cloud_point_ptrans.inv()

    # return the transform and the image coordinate used to generate the transform
    return [depth_pixel_T_endeffector_ptrans, pixel_coordinate_of_endeffector]


def endeffector_image_coordinate(camera_intrinsics_matrix, xyz, flip_x=1.0, flip_y=1.0):

    # get focal length and camera image center from the intrinsics matrix
    fx = camera_intrinsics_matrix[0, 0]
    fy = camera_intrinsics_matrix[1, 1]
    # center of image x coordinate
    center_x = camera_intrinsics_matrix[2, 0]
    # center of image y coordinate
    center_y = camera_intrinsics_matrix[2, 1]

    # Capital X is horizontal point, right in camera image frame
    X = xyz[0]
    # Capital Y is vertical point, up in camera image frame
    Y = xyz[1]
    # Capital Z is depth in camera frame
    Z = xyz[2]
    # x is the image coordinate horizontal axis
    x = flip_x * (X * fx / Z) + center_x
    # y is the image coordinate vertical axis
    y = flip_y * (Y * fy / Z) + center_y
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
    pixel_coordinate_of_endeffector = endeffector_image_coordinate(
        camera_intrinsics_matrix, cte_xyz)

    # The calculation of the gripper pose in the
    # image frame is done with the convention:
    # - X is right in the image frame
    # - Y is up in the image frame
    # - Z is depth
    XYZ = depth_image_pixel_to_cloud_point(depth_image,
                                           camera_intrinsics_matrix,
                                           pixel_coordinate_of_endeffector,
                                           augmentation_rectangle=augmentation_rectangle)

    # begin Temporary debug code
    import depth_image_encoding
    # flip image on image x axis center line
    # pixel_coordinate_of_endeffector[1] = depth_image.shape[1] - pixel_coordinate_of_endeffector[0]
    # pixel_coordinate_of_endeffector[0] = depth_image.shape[0] - pixel_coordinate_of_endeffector[1]
    XYZ_image = depth_image_encoding.depth_image_to_point_cloud(depth_image, camera_intrinsics_matrix)
    # TODO(ahundt) make sure rot180 + fliplr is applied upstream in the dataset and to the depth images, ensure consistency with image intrinsic
    test_XYZ = XYZ_image[int(pixel_coordinate_of_endeffector[0]), int(pixel_coordinate_of_endeffector[1]), :]

    print('XYZ: ', XYZ, ' test_XYZ:', test_XYZ, ' pixel_coordinate_of_endeffector: ', pixel_coordinate_of_endeffector, 'single_XYZ.shape: ', XYZ_image.shape)
    # assert(np.allclose(test_XYZ, XYZ))
    # end Temporary debug code
    return test_XYZ, pixel_coordinate_of_endeffector


def grasp_dataset_rotation_to_theta(rotation, verbose=0):
    """Convert a rotation to an angle theta specifically for brainrobotdata

    From above, a rotation to the right should be a positive theta,
    and a rotation to the left negative theta. The initial pose is with the
    z axis pointing down, the y axis to the right and the x axis forward.

    This format does not allow for arbitrary rotation commands to be defined,
    and originates from the paper and dataset:
    https://sites.google.com/site/brainrobotdata/home/grasping-dataset
    https://arxiv.org/abs/1603.02199

    In the google brain dataset the gripper is only commanded to
    rotate around a single vertical axis,
    so you might clearly visualize it, this also happens to
    approximately match the vector defined by gravity.
    Furthermore, the original paper had the geometry of the
    arm joints on which params could easily be extracted,
    which is not available here. To resolve this discrepancy
    Here we assume that the gripper generally starts off at a
    quaternion orientation of approximately [qx=-1, qy=0, qz=0, qw=0].
    This is equivalent to the angle axis
    representation of [a=np.pi, x=-1, y=0, z=0],
    which I'll name default_rot.

    It is also important to note the ambiguity of the
    angular distance between any current pose
    and the end pose. This angular distance will
    always have a positive value so the network
    could not naturally discriminate between
    turning left and turning right.
    For this reason, we use the angular distance
    from default_rot to define the input angle parameter,
    and if the angle axis x axis component is > 0
    we will use theta for rotation,
    but if the angle axis x axis component is < 0
    we will use -theta.
    """
    aa = eigen.AngleAxisd(rotation)
    theta = aa.angle()
    if aa.axis().z() < 0:
        multiply = 1.0
    else:
        multiply = -1.0
    if verbose > 0:
        print("ANGLE_AXIS_MULTIPLY: ", aa.angle(), np.array(aa.axis()), multiply)
    theta *= multiply
    return theta


def grasp_dataset_ptransform_to_vector_sin_theta_cos_theta(ptransform, dtype=np.float32):
    """Plucker transform to [dx, dy, dz, sin(theta), cos(theta)]
    Convert a PTransform 3D Rigid body transform into a numpy array with 5 total entries,
    including a 3 entry translation vector and 2 entries for
    a single rotation angle theta containing sin(theta), cos(theta). This format
    does not allow for arbitrary rotation commands to be defined,
    and originates from the paper and dataset:
    https://sites.google.com/site/brainrobotdata/home/grasping-dataset
    https://arxiv.org/abs/1603.02199

    In the google brain dataset the gripper is only commanded to
    rotate around a single vertical axis,
    so you might clearly visualize it, this also happens to
    approximately match the vector defined by gravity.
    Furthermore, the original paper had the geometry of the
    arm joints on which params could easily be extracted,
    which is not available here. To resolve this discrepancy
    Here we assume that the gripper generally starts off at a
    quaternion orientation of approximately [qx=-1, qy=0, qz=0, qw=0].
    This is equivalent to the angle axis
    representation of [a=np.pi, x=-1, y=0, z=0],
    which I'll name default_rot.

    It is also important to note the ambiguity of the
    angular distance between any current pose
    and the end pose. This angular distance will
    always have a positive value so the network
    could not naturally discriminate between
    turning left and turning right.
    For this reason, we use the angular distance
    from default_rot to define the input angle parameter,
    and if the angle axis x axis component is > 0
    we will use [sin(theta), cos(theta)] for rotation,
    but if the angle axis x axis component is < 0
    we will use [sin(-theta), cos(-theta)].

    Also note that in the eigen API being used:
    e.AngleAxisd(e.Quaterniond().Identity()) == [a=0.0, x=1, y=0, z=0]

    # Params

    ptransform: the PTransformd to convert

    # Returns

    vector_sin_theta_cos_theta
    """
    translation = np.squeeze(ptransform.translation())
    theta = grasp_dataset_rotation_to_theta(ptransform.rotation())
    sin_cos_theta = np.array([np.sin(theta), np.cos(theta)])
    vector_sin_theta_cos_theta = np.concatenate([translation, sin_cos_theta])
    vector_sin_theta_cos_theta = vector_sin_theta_cos_theta.astype(dtype)
    return vector_sin_theta_cos_theta


def grasp_dataset_to_ptransform(camera_T_base, base_T_endeffector):
    """Convert brainrobotdata features camera_T_base and base_T_endeffector to camera_T_endeffector and ptransforms.

    This specific function exists because it accepts the raw feature types
    defined in the google brain robot grasping dataset.

    # Params

    camera_T_base: a vector quaternion array
    base_T_endeffector: a 4x4 homogeneous 3D transformation matrix

    # Returns

      PTransformd formatted transforms:
      camera_T_endeffector_ptrans, base_T_endeffector_ptrans, base_T_camera_ptrans
    """
    base_T_endeffector_ptrans = vector_quaternion_array_to_ptransform(base_T_endeffector)
    # In this case camera_T_base is a transform that takes a point in the base
    # frame of reference and transforms it to the camera frame of reference.
    camera_T_base_ptrans = matrix_to_ptransform(camera_T_base)
    base_T_camera = camera_T_base_ptrans.inv()
    # Perform the actual transform calculation
    camera_T_endeffector_ptrans = base_T_endeffector_ptrans * base_T_camera.inv()
    return camera_T_endeffector_ptrans, base_T_endeffector_ptrans, base_T_camera


def current_endeffector_to_final_endeffector_feature(current_base_T_endeffector,
                                                     end_base_T_endeffector,
                                                     feature_type='vec_sin_cos_5',
                                                     dtype=np.float32):
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
    feature_type: String identifying the feature type to return, which should contain one of the following options:
       'vec_quat_7' A numpy array with 7 total entries including a 3 entry translation vector and 4 entry quaternion.
       'vec_sin_cos_5'  A numpy array with 5 total entries [dx, dy, dz, sin(theta), cos(theta)]
                        including a 3 entry translation vector and 2 entries for the angle of the rotation.
                       for a single rotation angle theta containing sin(theta), cos(theta). This format
                       does not allow for arbitrary commands to be defined, and originates from the paper and dataset:
                       https://sites.google.com/site/brainrobotdata/home/grasping-dataset
                       https://arxiv.org/abs/1603.02199

                       see also: grasp_dataset_ptransform_to_vector_sin_theta_cos_theta()
         'sin_cos_2' A numpy array with 2 total entries [sin(theta), cos(theta)]] for the angle of the rotation.

    # Returns

    A numpy array or object of the type specified in the feature_type parameter.

    """
    base_to_current = vector_quaternion_array_to_ptransform(current_base_T_endeffector)
    base_to_end = vector_quaternion_array_to_ptransform(end_base_T_endeffector)
    current_to_end = base_to_end * base_to_current.inv()

    # we have ptransforms for both data, now get transform from current to commanded
    if 'vec_quat_7' in feature_type:
        current_to_end = ptransform_to_vector_quaternion_array(current_to_end)
    elif 'vec_sin_cos_5' in feature_type:
        current_to_end = grasp_dataset_ptransform_to_vector_sin_theta_cos_theta(current_to_end)
    elif 'sin_cos_2' in feature_type:
        current_to_end = grasp_dataset_ptransform_to_vector_sin_theta_cos_theta(current_to_end)[-2:]
    else:
        raise ValueError('current_endeffector_to_final_endeffector_feature() '
                         'received unsupported feature type: ' + str(feature_type))
    return current_to_end.astype(dtype)


def grasp_dataset_to_surface_relative_transform_feature(
        depth_image,
        camera_intrinsics_matrix,
        camera_T_base,
        current_base_T_endeffector,
        end_base_T_endeffector,
        feature_type='delta_depth_sin_cos_3',
        augmentation_rectangle=None,
        dtype=np.float32):
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
    current_base_T_endeffector: A vector quaternion array from a base frame to an end effector frame
        vector (x, y, z) for cartesian motion and quaternion (qx, qy, qz, qw) for rotation.
        base_T_endeffector is a transform that takes a point in the endeffector
        frame of reference and transforms it to the base frame of reference at the current time step
        of the grasp attempt move_to_grasp sequence.
    end_base_T_endeffector: A vector quaternion array from a base frame to an end effector frame
        containing the proposed destination of the robot at the final time step of the grasp attempt
        move_to_grasp sequence.
    feature_type: String identifying the feature type to return, which should contain one of the following options:
        'delta_depth_sin_cos_3' [delta_depth, sin(theta), cos(theta)] where delta_depth depth offset for the gripper
            from the measured surface, alongside a single rotation angle theta containing sin(theta), cos(theta).
            This format does not allow for arbitrary commands to be defined, and the rotation component
            is based on the paper and dataset:
                        https://sites.google.com/site/brainrobotdata/home/grasping-dataset
                        https://arxiv.org/abs/1603.02199
        'delta_depth_quat_5' A numpy array with 5 total entries including depth offset and 4 entry quaternion.
            (Not yet implemented)
        'vec_quat_7' A numpy array with 7 total entries including a 3 entry translation vector and 4 entry quaternion.
            (Not yet implemented)
        'vec_sin_cos_5'  A numpy array with 5 total entries [dx, dy, dz, sin(theta), cos(theta)]
                        including a 3 entry translation vector and 2 entries for the angle of the rotation.
                        for a single rotation angle theta containing sin(theta), cos(theta). This format
                        does not allow for arbitrary commands to be defined, and originates from the paper and dataset:
                        https://sites.google.com/site/brainrobotdata/home/grasping-dataset
                        https://arxiv.org/abs/1603.02199

                        see also: grasp_dataset_ptransform_to_vector_sin_theta_cos_theta()
            (Not yet implemented)

    augmentation_rectangle:
       A random offset for the selected (dx, dy) pixel index.
       It will randomly select a pixel in a box around the endeffector coordinate.
       Default (1, 1) has no augmentation.

    # Returns

        [feature_array, depth_relative_vec_quat_array, image_coordinate,
            camera_T_endeffector_current_vec_quat_array,
            camera_T_endeffector_end_vec_quat_array, current_to_end_vec_quat_array]
        which will be in the following respective numpy formats (or their tf equivalents):
        [nptf.float32, np.float32, np.int32, np.float32, np.float32, np.float32]

        feature_array is a numpy array defined and described in the `feature_type` parameter.

        end_surface_relative_transform_pose is a numpy array pose [x, y, z, qx, qy, qz, qw],
        relative to the final time step  which contains:
           - vector (x, y, z) for cartesian motion
           - quaternion (qx, qy, qz, qw) for rotation

        The image_coordinate (dx, dy) is the pixel width, height
        coordinate of the transform in the depth image.

        image_coordinate is used to calculate the point cloud point used for the
        surface relative transform and to generate 2D label weights.
    """
    camera_T_endeffector_current_ptrans, _, _ = grasp_dataset_to_ptransform(camera_T_base, current_base_T_endeffector)
    camera_T_endeffector_end_ptrans, _, _ = grasp_dataset_to_ptransform(camera_T_base, end_base_T_endeffector)

    base_to_current = vector_quaternion_array_to_ptransform(current_base_T_endeffector)
    base_to_end = vector_quaternion_array_to_ptransform(end_base_T_endeffector)
    current_to_end = base_to_end * base_to_current.inv()
    current_to_end_vec_quat_array = ptransform_to_vector_quaternion_array(current_to_end)

    # calculate the surface relative transform
    depth_pixel_T_endeffector_ptrans, image_coordinate = surface_relative_transform(
        depth_image,
        camera_intrinsics_matrix,
        camera_T_endeffector_end_ptrans,
        augmentation_rectangle)

    # convert the transform into the vector + quaternion format
    depth_relative_vec_quat_array = ptransform_to_vector_quaternion_array(depth_pixel_T_endeffector_ptrans)
    camera_T_endeffector_current_vec_quat_array = ptransform_to_vector_quaternion_array(camera_T_endeffector_current_ptrans)
    camera_T_endeffector_end_vec_quat_array = ptransform_to_vector_quaternion_array(camera_T_endeffector_end_ptrans)

    # get the delta theta parameter
    sin_cos_theta = current_endeffector_to_final_endeffector_feature(
        current_base_T_endeffector,
        end_base_T_endeffector,
        feature_type='sin_cos_2',
        dtype=dtype)

    # get the delta depth offset
    # TODO(ahundt): verify that z correctly reflects the depth offset
    delta_depth = depth_pixel_T_endeffector_ptrans.translation().z()

    # create and return the user specified feature
    if feature_type == 'delta_depth_sin_cos_3':
        feature_array = np.concatenate([delta_depth, sin_cos_theta])
    if feature_type == 'delta_depth_quat_5':
        feature_array = np.concatenate([delta_depth, depth_relative_vec_quat_array[-4:]])
    else:
        raise TypeError('grasp_dataset_to_surface_relative_transform_feature()'
                        ' feature_type not yet supported: ' + str(feature_type))

    return [feature_array, depth_relative_vec_quat_array, image_coordinate,
            camera_T_endeffector_current_vec_quat_array,
            camera_T_endeffector_end_vec_quat_array, current_to_end_vec_quat_array]


def vector_quaternion_arrays_allclose(vq1, vq2, rtol=1e-6, atol=1e-6, verbose=0):
    """Check if all the entries are close for two vector quaternion numpy arrays.

    Quaterions are a way of representing rigid body 3D rotations that is more
    numerically stable and compact in memory than other methods such as a 3x3
    rotation matrix.

    This special comparison function is needed because for quaternions q == -q.
    Vector Quaternion numpy arrays are expected to be in format
    [x, y, z, qx, qy, qz, qw].

    # Params

    vq1: First vector quaternion array to compare.
    vq2: Second vector quaternion array to compare.
    rtol: relative tolerance.
    atol: absolute tolerance.

    # Returns

    True if the transforms are within the defined tolerance, False otherwise.
    """
    vq1 = np.array(vq1)
    vq2 = np.array(vq2)
    q3 = np.array(vq2[3:])
    q3 *= -1.
    v3 = vq2[:3]
    vq3 = np.array(np.concatenate([v3, q3]))
    comp12 = np.allclose(np.array(vq1), np.array(vq2), rtol=rtol, atol=atol)
    comp13 = np.allclose(np.array(vq1), np.array(vq3), rtol=rtol, atol=atol)
    if verbose > 0:
        print(vq1)
        print(vq2)
        print(vq3)
        print(comp12, comp13)
    return comp12 or comp13


def gaussian_kernel_2D(size=(3, 3), center=(1, 1), sigma=1):
    """Create a 2D gaussian kernel with specified size, center, and sigma.

    Output with the default parameters:

        [[ 0.36787944  0.60653066  0.36787944]
         [ 0.60653066  1.          0.60653066]
         [ 0.36787944  0.60653066  0.36787944]]

    references:

            https://stackoverflow.com/a/43346070/99379
            https://stackoverflow.com/a/32279434/99379

    To normalize:

        g = gaussian_kernel_2d()
        g /= np.sum(g)
    """
    xx, yy = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    kernel = np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2. * sigma ** 2))
    return kernel
