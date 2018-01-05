import tensorflow as tf
from grasp_model import tile_vector_as_image_channels
import keras
from keras import backend as K


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
    xx, yy = tf.meshgrid(tf.range(0, size[0]),
                         tf.range(0, size[1]),
                         indexing='xy')
    kernel = tf.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2. * sigma ** 2))
    return kernel


def grasp_segmentation_gaussian_loss(y_true, y_pred, gaussian_kernel_size=(3, 3), gaussian_sigma=1, loss=keras.losses.binary_crossentropy):
    """ Loss function incorporating grasp parameters.

    # Arguments

        y_true: is assumed to be [label, x_img_coord, y_image_coord]
        y_pred: is expected to be a 2D array of labels.
    """
    y_true_img = tile_vector_as_image_channels(y_true[0], y_pred)
    y_height_coordinate = y_true[1]
    x_width_coordinate = y_true[2]
    weights = gaussian_kernel_2D(gaussian_kernel_size, (y_height_coordinate, x_width_coordinate), gaussian_sigma)
    loss_img = loss(y_true_img, y_pred)
    weighted_loss_img = K.multiply(loss_img, weights)
    return K.sum(weighted_loss_img)


def grasp_segmentation_single_pixel_loss(y_true, y_pred, loss=keras.losses.binary_crossentropy):
    """ Applies loss_fn at a specific pixel coordinate

    # Arguments

        y_true: [ground_truth_label, y_height_coordinate, x_width_coordinate]
        y_pred: predicted values
    """
    label = y_true[0]
    y_height_coordinate = y_true[1]
    x_width_coordinate = y_true[2]
    gripper_coordinate_y_pred = y_pred[y_height_coordinate, x_width_coordinate]
    return loss(label, gripper_coordinate_y_pred)
