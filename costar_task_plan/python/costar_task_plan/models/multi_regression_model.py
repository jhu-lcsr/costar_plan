
import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
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

class RobotMultiFFRegression(AbstractAgentBasedModel):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        Read in taskdef for this model (or set of models). Use it to create the
        regression neural net that we can fit to compute our next action.

        Remember, here the "labels" are computed from the task model. We can
        use images and joint states together to compute next image or next
        joint state.
        '''

        super(RobotMultiFFRegression, self).__init__(*args, **kwargs)

        self.taskdef = taskdef
        self.model = None
        
        self.dropout_rate = 0.5
        
        self.img_dense_size = 512
        self.img_col_dim = 256
        self.img_num_filters = 32
        self.robot_col_dense_size = 128
        self.robot_col_dim = 64
        self.combined_dense_size = 64

    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            example, *args, **kwargs):
        '''
        Training data -- just direct regression based on MSE.
        '''

        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1

        img_ins, img_out = GetCameraColumn(
                img_shape,
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                self.img_dense_size,)
        robot_ins, robot_out = GetArmGripperColumns(
                arm_size, 
                gripper_size,
                self.robot_col_dim,
                self.dropout_rate,
                self.robot_col_dense_size,)

        x = Concatenate()([img_out, robot_out])
        x = Dense(self.combined_dense_size)(x)
        #x = Dense(self.combined_dense_size)(img_out)
        x = LeakyReLU(alpha=0.2)(x)
        arm_out = Dense(arm_size)(x)
        gripper_out = Dense(gripper_size)(x)

        model = Model(img_ins + robot_ins, [arm_out, gripper_out])
        #model = Model(img_ins, [arm_out])
        optimizer = self.getOptimizer()
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()

        self.model = model
        self.model.fit(
                x=[features, arm, gripper],
                y=[arm_cmd, gripper_cmd],
                epochs=self.epochs,
                batch_size=self.batch_size,
                )

    def save(self):
        if self.model is not None:
            self.model.save_weights(self.name + ".h5f")

    def load(self):
        pass
