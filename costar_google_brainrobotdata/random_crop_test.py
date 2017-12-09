import tensorflow as tf
import random_crop_parameters as rcp
import numpy as np


class random_crop_test(tf.test.TestCase):
    """Unit test for random_crop
    """
    def testRandomCrop(self):
        with self.test_session() as sess:
            input_size = tf.convert_to_tensor([10, 11, 3])
            cropped_size = tf.convert_to_tensor([2, 4, 3])
            test_input = tf.range(input_size[0] * input_size[1] * input_size[2])
            test_input = tf.reshape(test_input, input_size)
            offset = rcp.random_crop_offset(input_size, cropped_size)
            rcp_crop = rcp.crop_images(test_input, offset, cropped_size)
            tf_crop = tf.slice(test_input, offset, cropped_size)
            rcp_crop, tf_crop = sess.run([rcp_crop, tf_crop])

            self.assertAllEqual(rcp_crop, tf_crop)

    def testIntrinsics(self):
        with self.test_session() as sess:
            intrinsics_np = np.array([[225, 0, 0], [0, 225, 0], [125, 127, 1]])
            intrinsics_tf = tf.convert_to_tensor(intrinsics_np)
            offset_np = np.array([20, 23, 0])
            offset_tf = tf.convert_to_tensor(offset_np)
            cropped_intrinsics = rcp.crop_image_intrinsics(intrinsics_tf, offset_tf)
            intrinsics_tf = sess.run(cropped_intrinsics)

            intrinsics_np[2, 0] -= offset_np[0]
            intrinsics_np[2, 1] -= offset_np[1]

            self.assertAllEqual(intrinsics_np, intrinsics_tf)


if __name__ == '__main__':
    tf.test.main()
