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
from .sampler2 import *

from .dvrk import *


class ConditionalImageJigsaws(ConditionalImage):

    def __init__(self, *args, **kwargs):

        super(ConditionalImageJigsaws, self).__init__(*args, **kwargs)

        self.num_options = 16

    def _makePredictor(self, image):
        
        img_shape = image.shape[1:]

        img_in = Input(img_shape,name="predictor_img_in")
        img0_in = Input(img_shape,name="predictor_img0_in")
        option_in = Input((1,), name="predictor_option_in"))
        ins = [img0_in, img_in]

        if self.skip_connections:
            encoder = self._makeImageEncoder2(img_shape)
        else:
            encoder = self._makeImageEncoder(img_shape)

        if self.skip_connections:
            decoder = self._makeImageDecoder2(self.hidden_shape)
        else:
            decoder = self._makeImageDecoder(self.hidden_shape)

        # load encoder/decoder weights if found
        try:
            encoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                #"pretrain_image_gan_model",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
            decoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                #"pretrain_image_gan_model",
                "image_decoder.h5f"))
            decoder.trainable = self.retrain
        except Exception as e:
            if not self.retrain:
                raise e

        if self.skip_connections:
            h, s32, s16, s8 = encoder([img0_in, img_in])
        else:
            h = encoder(img_in)
            h0 = encoder(img0_in)

        # Create model for predicting image and label
        next_model = GetNextModel(h, self.num_options, 128,
                self.decoder_dropout_rate)
        next_model.compile(loss="mae", optimizer=self.getOptimizer())
        next_option_out = next_model([h0, h, option_in])
        self.next_model = next_model

        # create input for controlling noise output if that's what we decide
        # that we want to do
        if self.use_noise:
            z = Input((self.num_hypotheses, self.noise_dim))
            ins += [z]

        next_option_in = Input((1,), name="next_option_in")
        next_option_in2 = Input((1,), name="next_option_in2")
        ins += [next_option_in, next_option_in2]

        y = OneHot(self.num_options)(next_option_in)
        y = Flatten()(y)
        y2 = OneHot(self.num_options)(next_option_in2)
        y2 = Flatten()(y2)
        x = h
        tform = self._makeTransform()
        x = tform([h0, h, y])
        x2 = tform([h0, x, y2])
        image_out = decoder([x])
        image_out2 = decoder([x2])

        lfn = self.loss
        lfn2 = "logcosh"

        # =====================================================================
        # Create models to train
        predictor = Model(ins + [option_in],
                [image_out, image_out2, next_option_out])
        predictor.compile(
                loss=[lfn, lfn, "binary_crossentropy"],
                loss_weights=[1., 1., 0.1],
                optimizer=self.getOptimizer())
        train_predictor = Model(ins + [option_in], [image_out, image_out2])
        train_predictor.compile(
                loss=lfn, 
                optimizer=self.getOptimizer())
        return predictor, train_predictor, ins, h

    def _getData(self, image, label, goal_image, goal_label,
            prev_label, *args, **kwargs):
        '''
        Process a consecutive chunk of data, returning what we need
        '''

        I0 = image[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0, axis=0),[length,1,1,1]) 
        oin_1h = np.squeeze(self.toOneHot2D(label, self.num_options))
        if self.do_all:
            o1_1h = np.squeeze(self.toOneHot2D(o1, self.num_options))
            return [I0, I, o1, o2, oin], [ I_target, I_target2, o1_1h, v, qa, ga]
        else:
            return [I0, I, o1, o2, oin], [I_target, I_target2]


    def encode(self, obs):
        '''
        Encode available features into a new state

        Parameters:
        -----------
        [unknown]: all are parsed via _getData() function.
        '''
        return self.image_encoder.predict(obs[1])

    def decode(self, hidden):
        '''
        Decode features and produce a set of visualizable images or other
        feature outputs.

        '''
        return self.image_decoder.predict(hidden)

    def prevOption(self, features):
        '''
        Just gets the previous option from a set of features
        '''
        if self.use_noise:
            return features[4]
        else:
            return features[3]

    def encodeInitial(self, obs):
        '''
        Call the encoder but only on the initial image frame
        '''
        return self.image_encoder.predict(obs[0])

    def pnext(self, hidden, prev_option, features):
        '''
        Visualize based on hidden
        '''
        h0 = self.encodeInitial(features)

        print(self.next_model.inputs)
        #p = self.next_model.predict([h0, hidden, prev_option])
        p = self.next_model.predict([hidden, prev_option])
        #p = np.exp(p)
        #p /= np.sum(p)
        return p

    def value(self, hidden, prev_option, features):
        h0 = self.encodeInitial(features)
        #v = self.value_model.predict([h0, hidden, prev_option])
        v = self.value_model.predict([hidden, prev_option])
        return v

    def transform(self, hidden, option_in=-1):

        raise NotImplementedError('transform() not implemented')

    def act(self, *args, **kwargs):
        raise NotImplementedError('act() not implemented')

    def debugImage(self, features):
        return features[1]
