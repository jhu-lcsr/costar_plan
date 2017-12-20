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
from .sampler2 import *

class PretrainSampler(PredictionSampler2):

    def __init__(self, taskdef, *args, **kwargs):
        super(PretrainSampler, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = ImageCb

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, q_target, g_target,] = features
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        [tt, o1, v, qa, ga, I] = targets
        oin_1h = np.squeeze(self.toOneHot2D(oin, self.num_options))
        return [I0, I, q, g, oin], [I, q, g, oin_1h]

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''

        (images, arm, gripper) = features
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                images,
                arm,
                gripper)

        # =====================================================================
        # Load the image decoders
        img_in = Input(img_shape,name="predictor_img_in")
        img0_in = Input(img_shape,name="predictor_img0_in")
        encoder = self._makeImageEncoder(img_shape)

        try:
            encoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                "image_encoder.h5f"))
            encoder.trainable = False
        except Exception as e:
            pass

        enc = encoder([img0_in, img_in])
        if self.skip_connections:
            decoder = self._makeImageDecoder(self.hidden_shape,self.skip_shape)
        else:
            decoder = self._makeImageDecoder(self.hidden_shape)
        try:
            decoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                "image_decoder.h5f"))
            decoder.trainable = False
        except Exception as e:
            pass

        encoder.summary()
        decoder.summary()

        sencoder = self._makeStateEncoder(arm_size, gripper_size, False)
        #sencoder.load_weights(self._makeName(
        #    "pretrain_state_encoder_model", "state_encoder.h5f"))
        sdecoder = self._makeStateDecoder(arm_size, gripper_size,
                self.rep_channels)
        #sdecoder.load_weights(self._makeName(
        #    "pretrain_state_encoder_model", "state_decoder.h5f"))

        # =====================================================================
        # Load the arm and gripper representation
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        arm_gripper = Concatenate()([arm_in, gripper_in])
        label_in = Input((1,))
        ins = [img0_in, img_in, arm_in, gripper_in, label_in]

        # =====================================================================
        # combine these models together with state information and label
        # information
        hidden_encoder = self._makeToHidden(img_shape, arm_size, gripper_size, self.rep_size)
        if self.skip_connections:
            h, skip_rep = hidden_encoder(ins)
        else:
            h = hidden_encoder(ins)
        value_out, next_option_out = GetNextOptionAndValue(h,
                                                           self.num_options,
                                                           self.rep_size,
                                                           dropout_rate=0.5,
                                                           option_in=None)
        hidden_decoder = self._makeFromHidden(self.rep_size)
        if self.skip_connections:
            #img_x = hidden_decoder([x, skip_rep])
            img_x, arm_x, gripper_x, label_x = hidden_decoder([h, skip_rep])
        else:
            #img_x = hidden_decoder(x)
            hidden_decoder.summary()
            img_x, arm_x, gripper_x, label_x = hidden_decoder(h)
        ae_outs = [img_x, arm_x, gripper_x, label_x]
        ae2 = Model(ins, ae_outs)
        ae2.compile(
            loss=["mae","mae", "mae",
                "mae",],
            loss_weights=[1.,1.,.2,0.,],#0.25],
            optimizer=self.getOptimizer())
        ae2.summary()

        #return predictor, train_predictor, None, ins, enc
        return ae2, ae2, None, ins, enc


