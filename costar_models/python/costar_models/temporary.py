# -*- coding: utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

# pylint: disable=g-short-docstring-punctuation
"""Higher level ops for building layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages

def spatial_softmax(features,
                    temperature=None,
                    name=None,
                    variables_collections=None,
                    trainable=True,
                    data_format='NHWC'):
  """Computes the spatial softmax of a convolutional feature map.
  First computes the softmax over the spatial extent of each channel of a
  convolutional feature map. Then computes the expected 2D position of the
  points of maximal activation for each channel, resulting in a set of
  feature keypoints [x1, y1, ... xN, yN] for all N channels.
  Read more here:
  "Learning visual feature spaces for robotic manipulation with
  deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.
  Args:
    features: A `Tensor` of size [batch_size, W, H, num_channels]; the
      convolutional feature map.
    temperature: Softmax temperature (optional). If None, a learnable
      temperature is created.
    name: A name for this operation (optional).
    variables_collections: Collections for the temperature variable.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
  Returns:
    feature_keypoints: A `Tensor` with size [batch_size, num_channels * 2];
      the expected 2D locations of each channel's feature keypoint (normalized
      to the range (-1,1)). The inner dimension is arranged as
      [x1, y1, ... xN, yN].
  Raises:
    ValueError: If unexpected data_format specified.
    ValueError: If num_channels dimension is unspecified.
  """
  shape = array_ops.shape(features)
  static_shape = features.shape
  height, width, num_channels = shape[1], shape[2], static_shape[3]
  if num_channels.value is None:
    raise ValueError('The num_channels dimension of the inputs to '
                     '`spatial_softmax` should be defined. Found `None`.')

  with ops.name_scope(name, 'spatial_softmax', [features]) as name:
    print("name =", name)
    # Create tensors for x and y coordinate values, scaled to range [-1, 1].
    pos_x, pos_y = array_ops.meshgrid(math_ops.lin_space(-1., 1., num=height),
                                      math_ops.lin_space(-1., 1., num=width),
                                      indexing='ij')
    pos_x = array_ops.reshape(pos_x, [height * width])
    pos_y = array_ops.reshape(pos_y, [height * width])
    if temperature is None:
      temperature_collections = utils.get_variable_collections(
          variables_collections, name+'temperature')
      temperature = variables.model_variable(
          name+'temperature',
          shape=(),
          dtype=dtypes.float32,
          initializer=init_ops.ones_initializer(),
          collections=temperature_collections,
          trainable=trainable)
      # We assume all ops are [NBATCH, HEIGHT, WIDTH, CHANNELS] but this code
      # does not! It will reorder them appropriately.
      features = array_ops.reshape(
          array_ops.transpose(features, [0, 3, 1, 2]), [-1, height * width])

    softmax_attention = nn.softmax(features/temperature)
    expected_x = math_ops.reduce_sum(
        pos_x * softmax_attention, [1], keep_dims=True)
    expected_y = math_ops.reduce_sum(
        pos_y * softmax_attention, [1], keep_dims=True)
    expected_xy = array_ops.concat([expected_x, expected_y], 1)
    feature_keypoints = array_ops.reshape(
        expected_xy, [-1, num_channels.value * 2])
    feature_keypoints.set_shape([None, num_channels.value * 2])
    return feature_keypoints
