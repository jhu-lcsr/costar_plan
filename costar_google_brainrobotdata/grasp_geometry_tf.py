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

import tensorflow as tf
import tensorflow.python.keras.backend as K


def depth_image_to_point_cloud(depth, intrinsics_matrix):
    """Depth images become an XYZ point cloud in the camera frame with shape (depth.shape[0], depth.shape[1], 3).

    Transform a depth image into a point cloud in the camera frame with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    Based on:
    https://github.com/tensorflow/models/blob/master/research/cognitive_mapping_and_planning/src/depth_utils.py
    https://codereview.stackexchange.com/a/84990/10101

    # TODO(ahundt) move depth image creation into tensorflow ops

    # Arguments

      depth: is a 2-D ndarray with shape (rows, cols) containing
      32bit floating point depths in meters. The result is a 3-D array with
      shape (rows, cols, 3). Pixels with invalid depth in the input have
      NaN or 0 for the z-coordinate in the result.

      intrinsics_matrix: 3x3 matrix for projecting depth values to z values
      in the point cloud frame. http://ksimek.github.io/2013/08/13/intrinsic/

      transform: 4x4 Rt matrix for rotating and translating the point cloud
    """
    with tf.name_scope('depth_image_to_point_cloud'):
        # may need the following for indexing https://github.com/tensorflow/tensorflow/issues/206#issuecomment-338103956
        fx = intrinsics_matrix[0, 0]
        fy = intrinsics_matrix[1, 1]
        # center of image x coordinate
        center_x = intrinsics_matrix[2, 0]
        # center of image y coordinate
        center_y = intrinsics_matrix[2, 1]
        # TODO(ahundt) make sure rot90 + fliplr is applied upstream in the dataset and to the depth images, ensure consistency with image intrinsics
        depth = tf.image.flip_left_right(tf.image.rot90(depth, 3))
        depth_shape = K.int_shape(depth)
        x, y = tf.meshgrid(tf.arange(depth_shape[0]),
                           tf.arange(depth_shape[1]),
                           indexing='ij')
        X = (x - center_x) * depth / fx
        Y = (y - center_y) * depth / fy
        XYZ = tf.stack((tf.keras.backend.flatten(X),
                        tf.keras.backend.flatten(Y),
                        tf.keras.backend.flatten(depth))
                       ).reshape(depth_shape + (3,))

        return XYZ