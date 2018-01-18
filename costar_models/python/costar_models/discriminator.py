from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate, Multiply
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential

from .multi_sampler import *

class Discriminator(RobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(Discriminator, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = None

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, arm, gripper) = features
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                images,
                arm,
                gripper)

        disc = self._makeImageEncoder(img_shape, disc=True)
        disc.summary()
   
        return None, disc, None, None, None

    def _getData(self, *args, **kwargs):
        features, targets = GetAllMultiData(self.num_options, *args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        o1 = targets[1]
        oin_1h = np.squeeze(ToOneHot2D(oin, self.num_options))
        return [I0, I], [oin_1h]

class HuskyDiscriminator(RobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(HuskyDiscriminator, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = None
        self.num_options = 5

    def _makeModel(self, image, *args, **kwargs):
        '''
        Create model to predict possible manipulation goals.
        '''
        img_shape = image.shape[1:]
        disc = self._makeImageEncoder(img_shape, disc=True)
        disc.summary()

        self.train_predictor = disc

    def _getData(self, image, *args, **kwargs):
        I = np.array(image) / 255.
        return [I], [I]
