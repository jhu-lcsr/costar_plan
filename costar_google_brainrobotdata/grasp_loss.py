import tensorflow as tf
from grasp_model import tile_vector_as_image_channels
import keras
from keras import backend as K
from keras_contrib.losses import segmentation_losses


def gripper_coordinate_y_pred(y_true, y_pred):
    """ Get the predicted value at the coordinate found in y_true.

    # Arguments

        y_true: [ground_truth_label, y_height_coordinate, x_width_coordinate]
            Shape of y_true is [batch_size, 3].
        y_pred: Predicted values with shape [batch_size, img_height, img_width, 1].
    """
    with K.name_scope(name="gripper_coordinate_y_pred") as scope:
        yx_coordinate = tf.cast(y_true[:, 1:], tf.int32)
        yx_shape = K.int_shape(yx_coordinate)
        sample_index = K.expand_dims(tf.range(yx_shape[0]), axis=-1)
        byx_coordinate = K.concatenate([sample_index, yx_coordinate], axis=-1)

        # maybe need to transpose yx_coordinate?
        gripper_coordinate_y_pred = tf.gather_nd(y_pred, byx_coordinate)
        return gripper_coordinate_y_pred


def gripper_coordinate_y_true(y_true, y_pred=None):
    """ Get the label found in y_true which also contains coordinates.

    # Arguments

        y_true: [ground_truth_label, y_height_coordinate, x_width_coordinate]
            Shape of y_true is [batch_size, 3].
        y_pred: Predicted values with shape [batch_size, img_height, img_width, 1].
    """
    with K.name_scope(name="gripper_coordinate_y_true") as scope:
        label = tf.cast(y_true[:, :1], tf.float32)
        return label


def gaussian_kernel_2D(size=(3, 3), center=None, sigma=1):
    """Create a 2D gaussian kernel with specified size, center, and sigma.

    All coordinates are in (y, x) order, which is (height, width),
    with (0, 0) at the top left corner.

    Output with the default parameters `size=(3, 3) center=None, sigma=1`:

        [[ 0.36787944  0.60653066  0.36787944]
         [ 0.60653066  1.          0.60653066]
         [ 0.36787944  0.60653066  0.36787944]]

    Output with parameters `size=(3, 3) center=(0, 1), sigma=1`:

        [[0.60653067 1.         0.60653067]
        [0.36787945 0.60653067 0.36787945]
        [0.082085   0.13533528 0.082085  ]]
    # Arguments

        size: dimensions of the output gaussian (height_y, width_x)
        center: coordinate of the center (maximum value) of the output gaussian, (height_y, width_x).
            Default of None will automatically be the center coordinate of the output size.
        sigma: standard deviation of the gaussian in pixels

    # References:

            https://stackoverflow.com/a/43346070/99379
            https://stackoverflow.com/a/32279434/99379

    # How to normalize

        g = gaussian_kernel_2d()
        g /= np.sum(g)
    """
    with K.name_scope(name='gaussian_kernel_2D') as scope:
        if center is None:
            center_y = K.reshape(size[0] / 2, [1, 1])
            center_x = K.reshape(size[1] / 2, [1, 1])
        else:
            # tuple does not support assignment
            center_y = K.reshape(center[0], [1, 1])
            center_x = K.reshape(center[1], [1, 1])

        yy, xx = tf.meshgrid(tf.range(0, size[0]),
                             tf.range(0, size[1]),
                             indexing='ij')
        yy = tf.cast(yy, tf.float32)
        xx = tf.cast(xx, tf.float32)
        center_y = tf.cast(center_y, tf.float32)
        center_x = tf.cast(center_x, tf.float32)
        sigma_tensor = tf.constant([[(2.0 * sigma ** 2)]], tf.float32)
        # tf.exp requires float16, float32, float64, complex64, complex128
        kernel = tf.exp(tf.div(-((xx - center_x) ** 2 + (yy - center_y) ** 2), sigma_tensor))
        kernel = tf.Print(kernel, [center_x, center_y, xx, yy, sigma_tensor], 'gaussian_tf')
        return kernel


def segmentation_gaussian_measurement(
        y_true,
        y_pred,
        gaussian_sigma=10,
        measurement=segmentation_losses.binary_crossentropy):
    """ Apply metric or loss measurement incorporating a 2D gaussian.

        Only works with batch size 1.
        Loop and call this function repeatedly over each sample
        to use a larger batch size.

    # Arguments

        y_true: is assumed to be [label, x_img_coord, y_image_coord]
        y_pred: is expected to be a 2D array of labels
            with shape [1, img_height, img_width, 1].
    """
    with K.name_scope(name='grasp_segmentation_gaussian_loss') as scope:
        label = y_true[0, 0]
        y_height_coordinate = y_true[0, 1]
        x_width_coordinate = y_true[0, 2]
        label = K.reshape(label, [1, 1])
        image_shape = tf.Tensor.get_shape(y_pred)
        y_true_img = tile_vector_as_image_channels(label, image_shape)
        y_true_img = tf.cast(y_true_img, tf.float32)
        loss_img = measurement(y_true_img, y_pred)
        y_pred_shape = K.int_shape(y_pred)
        if len(y_pred_shape) == 3:
            y_pred_shape = y_pred_shape[:-1]
        if len(y_pred_shape) == 4:
            y_pred_shape = y_pred_shape[1:3]
        weights = gaussian_kernel_2D(size=y_pred_shape, center=(y_height_coordinate, x_width_coordinate), sigma=gaussian_sigma)
        loss_img = K.flatten(loss_img)
        weights = K.flatten(weights)
        weighted_loss_img = tf.multiply(loss_img, weights)
        loss_sum = K.sum(weighted_loss_img)
        loss_sum = K.reshape(loss_sum, [1, 1])
        return loss_sum


def segmentation_gaussian_measurement_batch(
        y_true,
        y_pred,
        gaussian_sigma=10,
        measurement=segmentation_losses.binary_crossentropy):
    """ Apply metric or loss measurement to a batch of data incorporating a 2D gaussian.

        Only works with batch size 1.
        Loop and call this function repeatedly over each sample
        to use a larger batch size.

    # Arguments

        y_true: is assumed to be [label, x_img_coord, y_image_coord]
        y_pred: is expected to be a 2D array of labels
            with shape [1, img_height, img_width, 1].
    """
    with K.name_scope(name='segmentation_gaussian_measurement_batch') as scope:
        y_pred_shape = tf.Tensor.get_shape(y_pred)
        batch_size = y_pred_shape[0]
        y_true = tf.split(y_true, batch_size)
        y_pred = tf.split(y_pred, batch_size)
        results = []
        for y_true_img, y_pred_img in zip(y_true, y_pred):
            result = segmentation_gaussian_measurement(
                y_true=y_true_img, y_pred=y_pred_img,
                gaussian_sigma=gaussian_sigma,
                measurement=measurement
            )
            results = results + [result]
        results = tf.concat(results, axis=0)
        return results


def segmentation_gaussian_binary_crossentropy(
        y_true,
        y_pred,
        gaussian_sigma=10):
    with K.name_scope(name='segmentation_gaussian_binary_crossentropy') as scope:
        results = segmentation_gaussian_measurement_batch(
            y_true, y_pred,
            measurement=segmentation_losses.binary_crossentropy,
            gaussian_sigma=gaussian_sigma)
        return results


def segmentation_single_pixel_measurement(y_true, y_pred, measurement=keras.losses.binary_crossentropy, name=None):
    """ Applies metric or loss function at a specific pixel coordinate.

    # Arguments

        y_true: [ground_truth_label, y_height_coordinate, x_width_coordinate]
            Shape of y_true is [batch_size, 3].
        y_pred: Predicted values with shape [batch_size, img_height, img_width, 1].
    """
    if name is None:
        name = 'grasp_segmentation_single_pixel_measurement'
    with K.name_scope(name=name) as scope:
        label = gripper_coordinate_y_true(y_true)
        single_pixel_y_pred = gripper_coordinate_y_pred(y_true, y_pred)
        return measurement(label, single_pixel_y_pred)


def segmentation_single_pixel_binary_crossentropy(y_true, y_pred):
    return segmentation_single_pixel_measurement(y_true, y_pred, measurement=keras.losses.binary_crossentropy,
                                                 name='segmentation_single_pixel_binary_crossentropy')


def segmentation_single_pixel_binary_accuracy(y_true, y_pred, name=None):
    return segmentation_single_pixel_measurement(y_true, y_pred, measurement=keras.metrics.binary_accuracy,
                                                 name='segmentation_single_pixel_binary_accuracy')


def mean_pred(y_true, y_pred):
    """ mean predicted value metric

        useful for detecting perverse
        conditions such as
        100% grasp_success == True
    """
    return K.mean(y_pred)


def mean_pred_single_pixel(y_true, y_pred):
    """ mean predicted value metric at individual pixel coordinates.

    useful for detecting perverse
    conditions such as
    100% grasp_success == True
    """
    with K.name_scope(name='mean_pred_single_pixel') as scope:
        single_pixel_y_pred = gripper_coordinate_y_pred(y_true, y_pred)
        return K.mean(single_pixel_y_pred)


def mean_true(y_true, y_pred):
    """ mean ground truth value metric

    useful for determining
    summary statistics when using
    the multi-dataset loader

    # Arguments

        y_true: [ground_truth_label, y_height_coordinate, x_width_coordinate]
            Shape of y_true is [batch_size, 3], or [ground_truth_label] with shape [batch_size].
        y_pred: Predicted values with shape [batch_size, img_height, img_width, 1].
    """
    with K.name_scope(name='mean_true') as scope:
        if len(K.int_shape(y_true)) == 2 and K.int_shape(y_true)[1] == 3:
            y_true = tf.cast(y_true[:, :1], tf.float32)
        return K.mean(y_true)
