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
from .multi import *

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

        disc = MakeImageClassifier(self, img_shape)
        disc.summary()
   
        return None, disc, None, None, None

    def _getData(self, *args, **kwargs):
        features, targets = GetAllMultiData(self.num_options, *args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        return [I0, I_target], [o1_1h]

class HuskyDiscriminator(RobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(HuskyDiscriminator, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = None
        self.num_options = HuskyNumOptions()

    def _makeModel(self, image, *args, **kwargs):
        '''
        Create model to predict possible manipulation goals.
        '''
        img_shape = image.shape[1:]
        disc = self._makeImageEncoder(img_shape, disc=True)
        disc.summary()

        self.train_predictor = disc

    def _getData(self, image, label, *args, **kwargs):
        I = np.array(image) / 255.
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        o1 = np.array(label)
        o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
        return [I0, I], o1_1h

