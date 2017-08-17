import keras
from keras.applications.resnet50 import ResNet50
from keras import backend as K

def grasp_model(clear_view_image_op, current_time_image_op, pose_op):
    clear_view_resnet50 = ResNet50(include_top=False, tensor=clear_view_image_op)
    current_time_resnet50 = ResNet50(include_top=False, tensor=clear_view_image_op)

    pose_filter = K.transpose(pose_op)
    tiled_pose_filter = K.tile(pose_filter, clear_view_resnet50.size)
    tiled_pose_filter = K.reshape(tiled_filter, (clear_view.shape[0], clear_view.shape[1], pose_filter.size))
    fused_data = K.concat([clear_view_resnet50, tiled_pose_filter, current_time_resnet50])

    grasp_prediction = ResNet50(fused_data, weights=None, include_top=True, classes=2)
    return grasp_prediction


def grasp_model_segmentation(clear_view_image_op, current_time_image_op, pose_op):
    raise NotImplementedError