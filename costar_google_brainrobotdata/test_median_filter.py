import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import median_filter
from grasp_median_filter import median_filter_tf


class MedianFilterTest(tf.test.TestCase):
    """Unit test for median filter use py_func
    """
    def testSquare(self):
        with self.test_session() as sess:
            test_input = np.random.random((10, 9)).astype(np.float32)
            test_kernel = (3, 3)
            test_input_tf = tf.convert_to_tensor(test_input)
            filter_result_tf = median_filter_tf(test_input_tf, test_kernel)
            filter_result_np = median_filter(test_input, test_kernel)
            filter_result_tf = sess.run(filter_result_tf)
            self.assertAllEqual(filter_result_tf, filter_result_np)

if __name__ == '__main__':
    tf.test.main()
