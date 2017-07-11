
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

class RobotMultiFFRegression(AbstractAgentBasedModel):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        Read in taskdef for this model (or set of models). Use it to create the
        regression neural net that we can fit to compute our next action.

        Remember, here the "labels" are computed from the task model. We can
        use images and joint states together to compute next image or next
        joint state.
        '''

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

    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            example, *args, **kwargs):
        '''
        Training data -- just direct regression.
        '''
        pass
