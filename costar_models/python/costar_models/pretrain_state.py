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

class PretrainStateAutoencoder(RobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PretrainStateAutoencoder, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = StateCb

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, arm, gripper) = features
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                images,
                arm,
                gripper)

        arm_in = Input((arm_size,), name="predictor_arm_in")
        gripper_in = Input((gripper_size,), name="predictor_gripper_in")
        option_in = Input((1,), name="predictor_option_in")
        ins = [arm_in, gripper_in, option_in]

        encoder = self._makeStateEncoder(arm_size, gripper_size, False)
        decoder = self._makeStateDecoder(arm_size, gripper_size)

        encoder.summary()
        decoder.summary()

        h = encoder(ins)
        out = decoder(h)

        ae = Model(ins, out)
        ae.compile(
                loss=self.loss,#[self.loss,self.loss,"categorical_crossentropy"],
                loss_weights=[1.,0.2],#,0.01],
                optimizer=self.getOptimizer())
        ae.summary()
    
        return ae, ae, None, ins, h

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, q_target, g_target,] = features
        oin_1h = np.squeeze(self.toOneHot2D(oin, self.num_options))
        return [q, g, oin], [q, g]#, oin_1h]
        #return [q, g], [q, g]
