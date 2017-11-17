import tensorflow as tf
import random_crop_parameters as rcp


class random_crop_test(tf.test.TestCase):
    """Unit test for random_crop
    """
    def testSquare(self):
        with self.test_session():
            test_input = tf.range(1, 10)
            test_input = tf.reshape(test_input, [1, -1])
            test_input = tf.tile(test_input, [10, 1])
            offset = rcp.random_crop_parameters([10, 10], [2, 4])
            rcp_crop = tf.slice(test_input, offset, [2, 4])
            # rcp.crop_images(test_input, [2, 4], offset)
            tf_crop = tf.slice(test_input, offset, [2, 4])
            self.assertAllEqual(tf_crop.eval(), rcp_crop.eval())

if __name__ == '__main__':
    tf.test.main()
