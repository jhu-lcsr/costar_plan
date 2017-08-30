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
import tensorflow as tf


from keras.engine import Layer


def tile_vector_as_image_channels(vector_op, image_shape):
    ivs = K.shape(vector_op)
    vector_op = K.reshape(vector_op, [ivs[0], 1, 1, ivs[1]])
    vector_op = K.tile(vector_op, K.stack([1, image_shape[1], image_shape[2], 1]))
    return vector_op


def grasp_model(clear_view_image_op,
                current_time_image_op,
                input_vector_op,
                input_image_shape=[512, 640, 3],
                input_vector_op_shape=[5],
                batch_size=11):
    print('input_vector_op pre tile: ', input_vector_op)

    input_vector_op = tile_vector_as_image_channels(input_vector_op, K.shape(clear_view_image_op))

    combined_input_data = tf.concat([clear_view_image_op, input_vector_op, current_time_image_op], -1)
    combined_input_shape = input_image_shape
    # add up the total number of channels
    combined_input_shape[-1] = combined_input_shape[-1] * 2 + input_vector_op_shape[0]
    growth_rate = 12
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
                     include_top=True,
                     input_tensor=combined_input_data,
                     activation='sigmoid',
                     classes=1,
                     weights=None,
                     #  nb_filter=nb_filters,
                     growth_rate=12,
                     reduction=0.5,
                     nb_dense_block=4)
    return model


def grasp_model_deleteme(clear_view_image_op=None,
                         current_time_image_op=None,
                         input_vector_op=None,
                         input_image_shape=[512, 640, 3],
                         input_vector_op_shape=[7]):
    if clear_view_image_op is not None:
        print('clear_view_image_op.shape: ', clear_view_image_op.shape)
    # TODO(ahundt) clear view only changes very occasionally compared to others, save output as variable.
    clear_view_resnet50 = ResNet50(include_top=False, input_tensor=clear_view_image_op, input_shape=input_image_shape)
    current_time_resnet50 = ResNet50(include_top=False, input_tensor=current_time_image_op, input_shape=input_image_shape)
    vect_input = Input(tensor=input_vector_op)
    clear_view_logits = Input(tensor=clear_view_resnet50.output)
    current_time_logits = Input(tensor=current_time_resnet50.output)

    # vec_image = Lambda(repeat_vector_as_image)([input_vector_op, current_time_resnet50.output])
    # vec_image = repeat_vector_as_image2(vect_input, current_time_resnet50.output)

    print(current_time_resnet50.output.shape)
    print(current_time_resnet50.output)
    print('clear_view_logits.shape: ', clear_view_logits.shape)
    print('clear_view_logits._keras_shape:', clear_view_logits._keras_shape)
    vec_size = 7
    vect_input = Reshape([11, 1, 1, vec_size])(vect_input)
    tile_width = 2
    tile_height = 2
    tile_shape = [1, 1, tile_width, tile_height, 1]

    vect_input_image = Lambda(lambda x: K.tile(x, tile_shape))(vect_input)
    # vec_image = repeat_vector_as_image4(vect_input, clear_view_logits)

    # TODO(ahundt) concatenate pose op at every pixel
    # pose_tensor = Input(tensor=input_vector_op)
    # Tile(pose_tensor, current_time_resnet50.output.shape)
    # pose_filter = K.transpose(input_vector_op)
    # tiled_pose_filter = K.tile(pose_filter, clear_view_resnet50.output.shape)
    # tiled_pose_filter = K.reshape(tiled_pose_filter, (clear_view.shape[0], clear_view.shape[1], pose_filter.size))
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    fused_data = Concatenate(concat_axis)([clear_view_resnet50.output,
                                           vect_input_image,
                                           current_time_resnet50.output])

    x = ResNet50(fused_data, weights=None)

    x = Flatten()(x.output)
    classes = 1  # single class: grasp_success
    x = Dense(classes, activation='sigmoid', name='fc1')(x)
    grasp_model = Model([clear_view_image_op, current_time_image_op, input_vector_op], x, name="grasp_model")
    return grasp_model


def grasp_model_segmentation(clear_view_image_op=None,
                             current_time_image_op=None,
                             input_vector_op=None,
                             input_image_shape=[512, 640, 3],
                             input_vector_op_shape=[5],
                             batch_size=11):

    if input_vector_op is not None:
        ims = tf.shape(clear_view_image_op)
        ivs = tf.shape(input_vector_op)
        input_vector_op = tf.reshape(input_vector_op, [1, 1, 1, ivs[0]])
        input_vector_op = tf.tile(input_vector_op, tf.stack([ims[0], ims[1], ims[2], ivs[0]]))

    combined_input_data = tf.concat([clear_view_image_op, input_vector_op, current_time_image_op], -1)
    combined_input_shape = input_image_shape
    combined_input_shape[-1] = combined_input_shape[-1] * 2 + input_vector_op_shape[0]
    model = DenseNetFCN(input_shape=combined_input_shape, include_top='global_average_pooling', input_tensor=combined_input_data, activation='sigmoid')
    return model