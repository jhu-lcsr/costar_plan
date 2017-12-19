import numpy as np
import tensorflow as tf


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
    with tf.name_scope('gaussian_kernel_2D'):
        sx, sy = tf.split(tf.convert_to_tensor(size), [1])
        cx, cy = tf.split(tf.convert_to_tensor(center), [1])
        sigma = tf.convert_to_tensor(sigma)
        xx, yy = tf.meshgrid(tf.keras.backend.arange(size[0]), tf.keras.backend.arange(size[1]))
        kernel = tf.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2. * sigma ** 2))
        return kernel
