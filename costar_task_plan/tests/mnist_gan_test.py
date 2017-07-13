#!/usr/bin/env python

import numpy as np
import unittest

from tensorflow.examples.tutorials.mnist import input_data

from costar_task_plan.models.gan import SimpleImageGan

class MNISTDCGANTest(unittest.TestCase):

    img_rows = 28
    img_cols = 28
    channels = 1
    num_labels = 10

    def test(self):
        x_train = input_data.read_data_sets("mnist",\
        	one_hot=True).train.images
        y_train = input_data.read_data_sets("mnist",\
        	one_hot=True).train.labels
        x_train = x_train.reshape(-1, self.img_rows,\
        	self.img_cols, self.channels).astype(np.float32)
        y_train = y_train.reshape(-1, self.num_labels).astype(np.float32)
        self.assertEqual(x_train.shape[0], y_train.shape[0])

        noise_dim = 100
        gan = SimpleImageGan(
                img_rows=self.img_rows,
                img_cols=self.img_cols,
                channels=self.channels,
                noise_dim=noise_dim)
        gan.fit(x_train, y_train, batch_size=50)


if __name__ == '__main__':
    unittest.main()
