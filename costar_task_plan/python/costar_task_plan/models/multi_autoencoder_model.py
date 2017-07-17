
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

from robot_multi_models import *

class RobotMultiAutoencoder(AbstractAgentBasedModel):
    '''
    This is a version of the Autoencoder agent based model. It doesn't really inherit
    too much from that though.

    This model is designed to work with the "--features multi" option of the
    costar bullet sim. This includes multiple viewpoints of a scene, including
    camera input and others.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        """
        Read in task def for this model (or set of models).
        Set up the presets for this particular type 
        """

        self.taskdef = taskdef
        
        self.generator_dim = 128

        self.img_dense_size = 512
        self.img_col_dim = 256
        self.img_num_filters = 32

        self.dropout_rate = 0.5


        super(RobotMultiAutoencoder, self).__init__(*args, **kwargs)


    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            example, *args, **kwargs):
        '''
        Set up the imitation autoencoder to learn a model of what actions we expect
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

        print label

        # Set up the learning problem as:
        # Goal: f(img, arm, gripper) --> arm_cmd, gripper_cmd

        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1

        ins, enc = GetEncoder(img_shape,
                arm_size,
                gripper_size,
                self.generator_dim,
                self.dropout_rate,
                self.img_num_filters,
                self.img_dense_size,)
        rep, dec = GetDecoder(self.generator_dim,
                            img_shape,
                            arm_size,
                            gripper_size,
                            dropout_rate=self.dropout_rate,
                            filters=self.img_num_filters,)
        decoder = Model([rep], dec)
        self.model = Model(ins, decoder(enc))
        optimizer = self.getOptimizer()
        self.model.compile(loss="mae", optimizer=optimizer)
        self.model.summary()
        self.model.fit(
                x=[features, arm, gripper],
                #y=[arm_cmd, gripper_cmd],
                y=[features],
                epochs=self.epochs,
                batch_size=self.batch_size,
                )       

