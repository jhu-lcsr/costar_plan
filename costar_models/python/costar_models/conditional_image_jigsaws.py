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

from .conditional_image import ConditionalImage
from .dvrk import *

class ConditionalImageJigsaws(ConditionalImage):

    def __init__(self, *args, **kwargs):

        super(ConditionalImageJigsaws, self).__init__(*args, **kwargs)

        self.num_options = 16

    def _makePredictor(self, image):

        img_shape = image.shape[1:]

        img0_in = Input(img_shape, name="predictor_img0_in")
        img_in = Input(img_shape, name="predictor_img_in")
        option_in = Input((1,), name="predictor_option_in")
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
                "pretrain_image_encoder_model_jigsaws",
                #"pretrain_image_gan_model",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
            decoder.load_weights(self._makeName(
                "pretrain_image_encoder_model_jigsaws",
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

        goal_image2, _ = GetNextGoal(goal_image, label)

        # Extend image_0 to full length of sequence
        image0 = image[0,:,:,:]
        length = image.shape[0]
        image0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1])

        return [image0, image, label, goal_label], [goal_image, goal_image2]

