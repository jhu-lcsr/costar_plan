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

from .sampler2 import *
from .multi import *
from .data_utils import *

class PretrainSampler(PredictionSampler2):

    def __init__(self, taskdef, *args, **kwargs):
        super(PretrainSampler, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = ImageCb

    def _getData(self, *args, **kwargs):
        features, targets = GetAllMultiData(self.num_options, *args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        [tt, o1, v, qa, ga, I_target] = targets
        oin_1h = np.squeeze(ToOneHot2D(oin, self.num_options))
        return [I, q, g, oin], [I, q, g, oin_1h]

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
        encoder = self._makeImageEncoder(img_shape)


        if self.skip_connections:
            decoder = self._makeImageDecoder(self.hidden_shape,self.skip_shape)
        else:
            decoder = self._makeImageDecoder(self.hidden_shape)

        try:
            decoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                "image_decoder.h5f"))
            encoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                "image_encoder.h5f"))
            decoder.trainable = self.retrain
            encoder.trainable = self.retrain
        except Exception as e:
            pass

        encoder.summary()
        decoder.summary()

        enc = encoder([img_in])
        rep_channels = self.encoder_channels
        sencoder = self._makeStateEncoder(arm_size, gripper_size, False)
        sdecoder = self._makeStateDecoder(arm_size, gripper_size,
                rep_channels)

        # =====================================================================
        # Load the arm and gripper representation
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        arm_gripper = Concatenate()([arm_in, gripper_in])
        label_in = Input((1,))
        ins = [img_in, arm_in, gripper_in, label_in]

        # =====================================================================
        # combine these models together with state information and label
        # information
        hidden_encoder = self._makeToHidden(img_shape, arm_size, gripper_size,
                rep_channels)
        if self.skip_connections:
            h, skip_rep = hidden_encoder(ins)
        else:
            h = hidden_encoder(ins)
        hidden_decoder = self._makeFromHidden(rep_channels)

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
            loss=["mae","mae", "mae", "categorical_crossentropy"],
            loss_weights=[1.,0.5,0.1,0.025,],
            optimizer=self.getOptimizer())
        ae2.summary()

        #return predictor, train_predictor, None, ins, enc
        return ae2, ae2, None, ins, enc

