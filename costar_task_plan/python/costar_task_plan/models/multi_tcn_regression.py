
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

class RobotMultiTCNFFRegression(AbstractAgentBasedModel):

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

    def _makeModel(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1

        num_models = 5

        ins, x = GetSeparateEncoder(
                img_shape=img_shape,
                img_col_dim=self.img_col_dim,
                img_dense_size=self.img_dense_size,
                arm_size=arm_size,
                gripper_size=gripper_size,
                dropout_rate=self.dropout_rate,
                img_num_filters=self.img_num_filters,
                robot_col_dim=self.robot_col_dim,
                combined_dense_size=self.combined_dense_size,
                robot_col_dense_size=self.robot_col_dense_size,)
           
        x.compile(loss="mse", optimizer=self.getOptimizer())
        x.summary()
        xlist = []
        for i in xrange(num_models):
                img = Input(img_shape)
                xlist.append(x[img, (arm_size,), (gripper_size,)])
                
        #arm_size,
        #gripper_size,
        #self.robot_col_dim,
        #self.robot_col_dense_size,
        #self.combined_dense_size)

        arm_out = Dense(arm_size)(x)
        gripper_out = Dense(gripper_size)(x)

        model = Model(ins, [arm_out, gripper_out])
        #model = Model(img_ins, [arm_out])
        optimizer = self.getOptimizer()
        model.compile(loss="mse", optimizer=optimizer)
        self.model = model

    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
        '''
        Training data -- just direct regression based on MSE from the other
        trajectory.
        '''

        self._makeModel(features, arm, gripper, arm_cmd,
                gripper_cmd, *args, **kwargs)
        self.model.summary()
        self.model.fit(
                x=[features, arm, gripper],
                y=[arm_cmd, gripper_cmd],
                epochs=self.epochs,
                batch_size=self.batch_size,
                )

    def predict(self, features):
        if isinstance(features, list):
            assert len(features) == len(self.model.inputs)
        if self.model is None:
            raise RuntimeError('model is missing')
        features = [f.reshape((1,)+f.shape) for f in features]
        res = self.model.predict(features)
        return res
