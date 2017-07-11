
import keras.backend as K
import keras.losses as losses
import numpy as np

from matplotlib import pyplot as plt

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam

from abstract import AbstractAgentBasedModel
from gan import GAN

class RobotMultiGAN(GAN):
    '''
    This is a version of the GAN agent based model. It doesn't really inherit
    too much from that though.

    This model is designed to work with the "--features multi" option of the
    costar bullet sim. This includes multiple viewpoints of a scene, including
    camera input and others.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        """
        Read in taskdef for this model (or set of models). Use it to create the
        generator and discriminator neural nets that we will be optimizing.
        """

        self.taskdef = taskdef
        
        img_rows = 768 / 8
        img_cols = 1024 / 8
        self.nchannels = 3

        self.img_shape = (img_rows, img_cols, self.nchannels)

        self.generator_dense_size = 1024
        self.generator_filters_c1 = 256

        self.discriminator_dense_size = 1024
        self.discriminator_filters_c1 = 512

        self.dropout_rate = 0.5

        """
        g_in, g_out, g_opt = self._generator(self.img_shape, labels, noise_dim)
        labels_input = g_in[-1]
        d_in, d_out, d_opt = self._discriminator(self.img_shape, labels,
                labels_input)

        super(RobotMultiGAN, self).__init__(
                [g_in, d_in],
                [g_out, d_out],
                [g_opt, d_opt],
                "binary_crossentropy",
                noise_dim)
        """

    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, actions,
            *args, **kwargs):
        '''
        Set up the imitation GAN to learn a model of what actions we expect
        from each state. Our goal is to sample the distribution of actions that
        is most likely to give us a trajectory to the goal.
        '''
        """
        imgs = data['features']
        arm = data['arm']
        gripper = data['gripper']
        arm_cmd = data['arm_cmd']
        gripper_cmd = data['gripper_cmd']
        labels = data['action']
        """

        print actions

        # Set up the learning problem as:
        # Goal: f(img, arm, gripper) --> arm_cmd, gripper_cmd

        inputs = [imgs, arm, gripper]
        targets = [arm_cmd, gripper_cmd]

        
