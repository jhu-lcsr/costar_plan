import tensorflow as tf

import keras
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import RepeatVector
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.merge import Concatenate
from keras.layers.merge import _Merge
from keras.models import Model
from keras.layers import Lambda
from keras.layers import Reshape
from keras.applications.imagenet_utils import _obtain_input_shape
from keras_contrib.applications.densenet import DenseNetFCN
from keras_contrib.applications.densenet import DenseNet

from keras.engine import Layer


def tile_vector_as_image_channels(vector_op, image_shape):
    """

    Takes a vector of length n and an image shape BHWC,
    and repeat the vector as channels at each pixel.
    """
    ivs = K.shape(vector_op)
    vector_op = K.reshape(vector_op, [ivs[0], 1, 1, ivs[1]])
    vector_op = K.tile(vector_op, K.stack([1, image_shape[1], image_shape[2], 1]))
    return vector_op


def grasp_model(clear_view_image_op,
                current_time_image_op,
                input_vector_op,
                input_image_shape=None,
                input_vector_op_shape=None,
                growth_rate=12,
                reduction=0.5,
                dense_blocks=4,
                include_top=True,
                dropout_rate=0.0):
    if input_vector_op_shape is None:
        input_vector_op_shape = [5]
    if input_image_shape is None:
        input_image_shape = [512, 640, 3]
    print('input_vector_op pre tile: ', input_vector_op)

    input_vector_op = tile_vector_as_image_channels(input_vector_op, K.shape(clear_view_image_op))

    combined_input_data = tf.concat([clear_view_image_op, input_vector_op, current_time_image_op], -1)
    combined_input_shape = input_image_shape
    # add up the total number of channels
    combined_input_shape[-1] = combined_input_shape[-1] * 2 + input_vector_op_shape[0]
    # initial number of filters should be
    # the number of input channels times the growth rate
    # nb_filters = combined_input_shape[-1] * growth_rate
    print('combined_input_shape: ', combined_input_shape)
    # print('nb_filters: ', nb_filters)
    print('combined_input_data: ', combined_input_data)
    print('clear_view_image_op: ', clear_view_image_op)
    print('current_time_image_op: ', current_time_image_op)
    print('input_vector_op: ', input_vector_op)
    model = DenseNet(input_shape=combined_input_shape,
                     include_top=include_top,
                     input_tensor=combined_input_data,
                     activation='sigmoid',
                     classes=1,
                     weights=None,
                     #  nb_filter=nb_filters,
                     growth_rate=growth_rate,
                     reduction=reduction,
                     nb_dense_block=dense_blocks,
                     dropout_rate=dropout_rate)
    return model


def grasp_model_segmentation(clear_view_image_op=None,
                             current_time_image_op=None,
                             input_vector_op=None,
                             input_image_shape=None,
                             input_vector_op_shape=None,
                             growth_rate=12,
                             reduction=0.5,
                             dense_blocks=4,
                             dropout_rate=0.0):
    if input_vector_op_shape is None:
        input_vector_op_shape = [5]
    if input_image_shape is None:
        input_image_shape = [512, 640, 3]

    if input_vector_op is not None:
        ims = tf.shape(clear_view_image_op)
        ivs = tf.shape(input_vector_op)
        input_vector_op = tf.reshape(input_vector_op, [1, 1, 1, ivs[0]])
        input_vector_op = tf.tile(input_vector_op, tf.stack([ims[0], ims[1], ims[2], ivs[0]]))

    combined_input_data = tf.concat([clear_view_image_op, input_vector_op, current_time_image_op], -1)
    combined_input_shape = input_image_shape
    combined_input_shape[-1] = combined_input_shape[-1] * 2 + input_vector_op_shape[0]
    model = DenseNetFCN(input_shape=combined_input_shape,
                        include_top='global_average_pooling',
                        input_tensor=combined_input_data,
                        activation='sigmoid',
                        growth_rate=growth_rate,
                        reduction=reduction,
                        nb_dense_block=dense_blocks)
    return model
