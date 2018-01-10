import tensorflow as tf
from grasp_model import tile_vector_as_image_channels
import keras
from keras import backend as K
from keras_contrib.losses import segmentation_losses


def gaussian_kernel_2D(size=(3, 3), center=(1, 1), sigma=1):
    """Create a 2D gaussian kernel with specified size, center, and sigma.

    Output with the parameters `size=(3, 3) center=(1, 1), sigma=1`:

        [[ 0.36787944  0.60653066  0.36787944]
         [ 0.60653066  1.          0.60653066]
         [ 0.36787944  0.60653066  0.36787944]]

    # References

            https://stackoverflow.com/a/43346070/99379
            https://stackoverflow.com/a/32279434/99379

    # To normalize

        g = gaussian_kernel_2d()
        g /= np.sum(g)
    """
    with K.name_scope(name='gaussian_kernel_2D') as scope:
        xx, yy = tf.meshgrid(tf.range(0, size[0]),
                             tf.range(0, size[1]),
                             indexing='xy')
        kernel = tf.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2. * sigma ** 2))
        return kernel


def segmentation_gaussian_measurement(
        y_true,
        y_pred,
        gaussian_kernel_size=(3, 3),
        gaussian_sigma=1,
        measurement=segmentation_losses.binary_crossentropy):
    """ Loss function incorporating grasp parameters.

    # Arguments

        y_true: is assumed to be [label, x_img_coord, y_image_coord]
        y_pred: is expected to be a 2D array of labels.
    """
    with K.name_scope(name='grasp_segmentation_gaussian_loss') as scope:
        label = y_true[0]
        y_height_coordinate = y_true[1]
        x_width_coordinate = y_true[2]

        y_true_img = tile_vector_as_image_channels(label, y_pred)
        #
        loss_img = (y_true_img, y_pred)
        y_pred_shape = K.int_shape(y_pred)
        if len(y_pred_shape) == 3:
            y_pred_shape = y_pred[:-1]
        if len(y_pred_shape) == 4:
            y_pred_shape = y_pred[1:3]
        weights = gaussian_kernel_2D(size=y_pred_shape, center=(y_height_coordinate, x_width_coordinate), sigma=gaussian_sigma)
        weighted_loss_img = tf.multiply(loss_img, weights)
        return K.sum(K.flatten(weighted_loss_img))


def segmentation_gaussian_binary_crossentropy(
        y_true,
        y_pred,
        gaussian_kernel_size=(3, 3),
        gaussian_sigma=1):
    return segmentation_gaussian_measurement(
        y_true=y_true, y_pred=y_pred,
        gaussian_kernel_size=gaussian_kernel_size,
        gaussian_sigma=gaussian_sigma,
        measurement=segmentation_losses.binary_crossentropy
    )


def segmentation_single_pixel_measurement(y_true, y_pred, measurement=keras.losses.binary_crossentropy, name=None):
    """ Applies metric or loss function at a specific pixel coordinate.

    # Arguments

        y_true: [ground_truth_label, y_height_coordinate, x_width_coordinate]
        y_pred: predicted values
    """
    if name is None:
        name = 'grasp_segmentation_single_pixel_measurement'
    with K.name_scope(name=name) as scope:
        label = tf.cast(y_true[:, :1], tf.float32)
        yx_coordinate = tf.cast(y_true[:, 1:], tf.int32)
        yx_shape = K.int_shape(yx_coordinate)
        sample_index = tf.expand_dims(tf.range(yx_shape[0]), axis=-1)
        byx_coordinate = tf.concat([sample_index, yx_coordinate], axis=-1)

        # maybe need to transpose yx_coordinate?
        gripper_coordinate_y_pred = tf.gather_nd(y_pred, byx_coordinate)
        return measurement(label, gripper_coordinate_y_pred)


def segmentation_single_pixel_binary_crossentropy(y_true, y_pred):
    return segmentation_single_pixel_measurement(y_true, y_pred, measurement=keras.losses.binary_crossentropy,
                                                 name='segmentation_single_pixel_binary_crossentropy')


def segmentation_single_pixel_binary_accuracy(y_true, y_pred, name=None):
    return segmentation_single_pixel_measurement(y_true, y_pred, measurement=keras.metrics.binary_accuracy,
                                                 name='segmentation_single_pixel_binary_accuracy')