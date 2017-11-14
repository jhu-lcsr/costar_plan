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
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from .abstract import *
from .callbacks import *
from .robot_multi_models import *
from .split import *
from .mhp_loss import *
from .loss import *
from .multi_sampler import *

class PretrainImageAutoencoder(RobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PretrainImageAutoencoder, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = ImageCb
        self.encoder_channels = 16

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, arm, gripper) = features
        img_shape = images.shape[1:]

        img0_in = Input(img_shape,name="predictor_img0_in")
        img_in = Input(img_shape,name="predictor_img_in")
        encoder = self._makeImageEncoder(img_shape)
        ins = [img0_in, img_in]
        if self.skip_connections:
            enc, skip = encoder(ins)
            decoder = self._makeImageDecoder(self.hidden_shape,(64,64,32))
            out = decoder([enc, skip])
        else:
            enc = encoder(ins)
            decoder = self._makeImageDecoder(self.hidden_shape)
            out = decoder(enc)

        encoder.summary()
        decoder.summary()

        ae = Model(ins, out)
        ae.compile(
                loss="mae",
                optimizer=self.getOptimizer())
        ae.summary()
    
        return ae, ae, None, [img_in], enc

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        I = features[0]
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        o1 = targets[1]
        return [I0, I], [I]
