import tensorflow as tf
from scipy.ndimage.filters import median_filter


def median_filter_tf(input_tensor, filter_size):
    """Median filter of tensor
       input_tensor is 2D tensor tf.float32
       filter_size is a tuple (x, y)
    """
    [filter_result] = tf.py_func(
        median_filter,
        [input_tensor, filter_size],
        [tf.float32], stateful=False,
        name='py_func/median_filter_tf')
    filter_result.set_shape(input_tensor.get_shape().as_list())
    filter_result = tf.reshape(filter_result, tf.shape(input_tensor))
    return filter_result
