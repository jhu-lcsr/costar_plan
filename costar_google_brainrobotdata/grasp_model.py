import tensorflow as tf

import keras
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
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

import keras_contrib
from keras_contrib.applications.densenet import DenseNetFCN
from keras_contrib.applications.densenet import DenseNet
from keras_contrib.applications.densenet import DenseNetImageNet121
from keras_contrib.applications.resnet import ResNet

from keras.engine import Layer


def tile_vector_as_image_channels(vector_op, image_shape):
    """
    Takes a vector of length n and an image shape BHWC,
    and repeat the vector as channels at each pixel.
    """
    with K.name_scope('tile_vector_as_image_channels'):
        ivs = K.shape(vector_op)
        print('input_vector_shape: ',ivs)
        # reshape the vector into a single pixel
        vector_pixel_shape = [ivs[0], 1, 1, ivs[1]]
        print('vector_pixel_shape: ', vector_pixel_shape)
        vector_op = K.reshape(vector_op, vector_pixel_shape)
        # tile the pixel into a full image
        # tile_dimensions = K.stack([1, image_shape[1], image_shape[2], 1])
        tile_dimensions = [1, image_shape[1], image_shape[2], 1]
        print('tile_dimensions to add: ', tile_dimensions)
        vector_op = K.tile(vector_op, tile_dimensions)
        print('tile_vector_as_image_channels default shape: ', vector_op)
        if K.backend() is 'tensorflow':
            output_shape = [ivs[0], image_shape[1], image_shape[2], ivs[1]]
            vector_op.set_shape(output_shape)
        print('tile_vector_as_image_channels with set shape vector_op: ', vector_op)
        return vector_op


def combine_images_with_tiled_vectors(images, vectors):
    with K.name_scope('combine_images_and_tile_vectors_as_image_channels'):
        if not isinstance(images, list):
            images = [images]
        if isinstance(vectors, list):
            # just concat all the vectors into a big one if needed
            vectors = K.concatenate(vectors)
        image_shape = images[0].get_shape().as_list()
        tiled_vectors = tile_vector_as_image_channels(vectors, image_shape)
        images.append(tiled_vectors)
        print('images and tiled vectors: ', images)
        combined = K.concatenate(images)

        print('combined concatenated images: ', combined)
        return combined


def tile_vector_as_image_channels_layer(images, vector, image_shape=None, vector_shape=None):
    """Tile a vector as if it were channels onto every pixel of an image

    # Params
       images: a list of images to combine, must have equal dimensions
       vector: the 1D vector to tile onto every pixel
       image_shape: Tuple with 3 entries defining the shape (batch, height, width)
           images should be expected to have, do not specify the number
           of batches.
       vector_shape: Tuple with 3 entries defining the shape (batch, height, width)
           images should be expected to have, do not specify the number
           of batches.
    """
    if not isinstance(images, list):
        images = [images]
    if vector_shape is None:
        # check if K.shape, K.int_shape, or vector.get_shape().as_list()[1:] is better
        # https://github.com/fchollet/keras/issues/5211
        vector_shape = K.shape(vector)[1:]
    if image_shape is None:
        # check if K.shape, K.int_shape, or image.get_shape().as_list()[1:] is better
        # https://github.com/fchollet/keras/issues/5211
        image_shape = K.shape(images[0])[1:]
    vector = Reshape([1, 1, vector_shape[-1]])(vector)
    tile_shape = (int(1), int(image_shape[0]), int(image_shape[1]), 1)
    tiled_vector = Lambda(lambda x: K.tile(x, tile_shape))(vector)
    x = Concatenate(axis=-1)([] + images + [tiled_vector])
    return x


def grasp_model_resnet(clear_view_image_op,
                       current_time_image_op,
                       input_vector_op,
                       input_image_shape=None,
                       input_vector_op_shape=None,
                       include_top=True,
                       dropout_rate=0.0):
    # if input_vector_op_shape is None:
    #     input_vector_op_shape = input_vector_op.get_shape().as_list()
    # if input_image_shape is None:
    #     input_image_shape = [512, 640, 3]
    print('input_vector_op pre tile: ', input_vector_op)
    print('clear_view_image_op pre tile: ', clear_view_image_op)
    print('current_time_image_op pre tile: ', current_time_image_op)

    # combined_input_data = tile_vector_as_image_channels_layer(
    #     [clear_view_image_op, current_time_image_op], input_vector_op, input_image_shape, input_vector_op_shape)

    combined_input_data = combine_images_with_tiled_vectors([clear_view_image_op, current_time_image_op], input_vector_op)

    if K.backend() is 'tensorflow':
        combined_input_shape = combined_input_data.get_shape().as_list()
    else:
        combined_input_shape = K.shape(combined_input_data)
    # tile_vector_as_image_channels(input_vector_op, clear_view_image_op.get_shape().as_list()):
    # combined_input_data = tf.concat([clear_view_image_op, input_vector_op, current_time_image_op], -1)
    # combined_input_shape = input_image_shape
    # add up the total number of channels
    # combined_input_shape[-1] = combined_input_shape[-1] * 2 + input_vector_op_shape[0]
    # combined_input_shape = K.shape(combined_input_data)
    print('combined_input_shape: ', combined_input_shape)
    # print('nb_filters: ', nb_filters)
    print('combined_input_data: ', combined_input_data)
    print('clear_view_image_op: ', clear_view_image_op)
    print('current_time_image_op: ', current_time_image_op)
    print('input_vector_op: ', input_vector_op)
    model = ResNet(input_shape=combined_input_shape,
                   classes=1,
                   block='bottleneck',
                   repetitions=[1, 1, 1, 1],
                   include_top=include_top,
                   input_tensor=combined_input_data,
                   activation='sigmoid',
                   initial_filters=96,
                   initial_kernel_size=(3, 3),
                   pooling=None,
                   dropout=dropout_rate)
    return model


def grasp_model_pretrained(clear_view_image_op,
                           current_time_image_op,
                           input_vector_op,
                           input_image_shape=None,
                           input_vector_op_shape=None,
                           growth_rate=12,
                           reduction=0.75,
                           dense_blocks=4,
                           include_top=True,
                           dropout_rate=0.0,
                           train_densenet=False):
    """export CUDA_VISIBLE_DEVICES="1" && python grasp_train.py --random_crop=1 --batch_size=1 --grasp_model grasp_model_pretrained --resize_width=320 --resize_height=256
    """
    if input_vector_op_shape is None:
        input_vector_op_shape = [K.shape(input_vector_op)[0], 7]
        input_vector_op = K.reshape(input_vector_op, input_vector_op_shape)
    if input_image_shape is None:
        input_image_shape = [512, 640, 3]

    print('input_image_shape:', input_image_shape)
    print('shape(input_image_shape:', input_image_shape)

    clear_view_model = ResNet50(
        input_shape=input_image_shape,
        input_tensor=clear_view_image_op,
        include_top=False)

    current_time_model = ResNet50(
        input_shape=input_image_shape,
        input_tensor=current_time_image_op,
        include_top=False)

    if not train_densenet:
        for layer in clear_view_model.layers:
            layer.trainable = False
        for layer in current_time_model.layers:
            layer.trainable = False

    clear_view_unpooled_layer = clear_view_model.layers[-2].get_output_at(0)
    unpooled_shape = clear_view_unpooled_layer.get_shape().as_list()
    print('clear_view_unpooled_layer: ', clear_view_unpooled_layer)
    print('unpooled_shape: ', unpooled_shape)
    current_time_unpooled = current_time_model.layers[-2].get_output_at(0)
    print('input_vector_op before tile: ', input_vector_op)
    input_vector_op = tile_vector_as_image_channels(
        input_vector_op,
        unpooled_shape
        )
    print('input_vector_op after tile: ', input_vector_op)
    print('clear_view_model.outputs: ', clear_view_model.outputs)
    print('current_time_model.outputs: ', current_time_model.outputs)
    combined_input_data = tf.concat([clear_view_unpooled_layer, input_vector_op, current_time_unpooled], -1)

    print('combined_input_data.get_shape().as_list():', combined_input_data.get_shape().as_list())
    combined_input_shape = combined_input_data.get_shape().as_list()
    combined_input_shape[-1] = unpooled_shape[-1] * 2 + input_vector_op_shape[-1]
    model_name = 'resnet'
    if model_name == 'dense':
        final_nb_layer = 4
        nb_filter = combined_input_shape[-1]
        weight_decay = 1e-4
        # The last dense_block does not have a transition_block
        x, nb_filter = keras_contrib.applications.densenet.__dense_block(
            combined_input_data, final_nb_layer, nb_filter, growth_rate,
            bottleneck=True, dropout_rate=dropout_rate, weight_decay=weight_decay)

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1, activation='sigmoid')(x)
    elif model_name == 'densenet':
        model = DenseNet(input_shape=combined_input_shape[1:],
                         include_top=include_top,
                         input_tensor=combined_input_data,
                         activation='sigmoid',
                         classes=1,
                         nb_filter=int(combined_input_shape[-1]*2),
                         growth_rate=growth_rate,
                         reduction=reduction,
                         nb_dense_block=dense_blocks,
                         dropout_rate=dropout_rate,
                         nb_layers_per_block=[6, 12, 24, 16],
                         subsample_initial_block=False,
                         weight_decay=1e-4,
                         pooling=None,
                         bottleneck=True)
    elif model_name == 'resnet':
        print('combined_input_shape: ', combined_input_shape)
        print('combined_input_data: ', combined_input_data)
        model = ResNet(input_shape=combined_input_shape[1:],
                       classes=1,
                       block='bottleneck',
                       repetitions=[1, 1, 1, 1],
                       include_top=include_top,
                       input_tensor=combined_input_data,
                       activation='sigmoid',
                       initial_filters=96,
                       initial_kernel_size=(3, 3),
                       pooling=None,
                       dropout=dropout_rate)

    return model


def grasp_model(clear_view_image_op,
                current_time_image_op,
                input_vector_op,
                input_image_shape=None,
                input_vector_op_shape=None,
                depth=40,
                growth_rate=36,
                reduction=0.5,
                dense_blocks=3,
                include_top=True,
                dropout_rate=0.0):
    if input_vector_op_shape is None:
        input_vector_op_shape = input_vector_op.get_shape().as_list()
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
                     depth=depth,
                     activation='sigmoid',
                     classes=1,
                     weights=None,
                     #  nb_filter=nb_filters,
                     growth_rate=growth_rate,
                     reduction=reduction,
                     nb_dense_block=dense_blocks,
                     dropout_rate=dropout_rate,
                     bottleneck=True)
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
        input_vector_op_shape = input_vector_op.get_shape().as_list()
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
