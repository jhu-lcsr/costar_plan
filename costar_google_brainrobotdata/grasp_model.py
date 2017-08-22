import keras
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.merge import Concatenate
from keras.models import Model


def grasp_model(clear_view_image_op, current_time_image_op, pose_op):
    print('clear_view_image_op.shape: ', clear_view_image_op.shape)
    clear_view_resnet50 = ResNet50(include_top=False, input_tensor=clear_view_image_op)
    current_time_resnet50 = ResNet50(include_top=False, input_tensor=current_time_image_op)

    # TODO(ahundt) concatenate pose op at every pixel
    # pose_tensor = Input(tensor=pose_op)
    # Tile(pose_tensor, current_time_resnet50.output.shape)
    # pose_filter = K.transpose(pose_op)
    # tiled_pose_filter = K.tile(pose_filter, clear_view_resnet50.output.shape)
    # tiled_pose_filter = K.reshape(tiled_pose_filter, (clear_view.shape[0], clear_view.shape[1], pose_filter.size))
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    fused_data = Concatenate(concat_axis)([clear_view_resnet50.output, current_time_resnet50.output])

    x = ResNet50(fused_data, weights=None)

    x = Flatten()(x.output)
    classes = 1  # single class: grasp_success
    x = Dense(classes, activation='sigmoid', name='fc1')(x)
    grasp_model = Model([clear_view_image_op, current_time_image_op, pose_op], x, name="grasp_model")
    return grasp_model


def grasp_model_segmentation(clear_view_image_op, current_time_image_op, pose_op):
    raise NotImplementedError