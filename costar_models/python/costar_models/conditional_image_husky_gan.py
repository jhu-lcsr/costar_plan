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

from .conditional_image_gan import *

class ConditionalImageHuskyGan(ConditionalImageGan):
    def __init__(self, *args, **kwargs):
        super(ConditionalImageHuskyGan, self).__init__(*args, **kwargs)

    def _getData(self, *args, **kwargs):
        return GetConditionalHuskyData(self.do_all, self.num_options, *args, **kwargs)

    def _makeModel(self, image, pose, *args, **kwargs):


        img_shape = image.shape[1:]
        pose_size = pose.shape[-1]

        # =====================================================================
        # Load the image decoders
        img_in = Input(img_shape,name="predictor_img_in")
        img0_in = Input(img_shape,name="predictor_img0_in")
        label_in = Input((1,))
        ins = [img0_in, img_in]

        encoder = self._makeImageEncoder(img_shape)
        try:
            encoder.load_weights(self._makeName(
                #"pretrain_image_encoder_model_husky",
                "pretrain_image_gan_model_husky",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
        except Exception as e:
            if not self.retrain:
                raise e

        decoder = self._makeImageDecoder(self.hidden_shape)
        try:
            decoder.load_weights(self._makeName(
                "pretrain_image_encoder_model_husky",
                #"pretrain_image_gan_model",
                "image_decoder.h5f"))
            decoder.trainable = self.retrain
        except Exception as e:
            if not self.retrain:
                raise e

        # =====================================================================
        # Load the arm and gripper representation
        if self.skip_connections:
            h, s32, s16, s8 = encoder([img0_in, img_in])
        else:
            h = encoder([img_in])
            h0 = encoder(img0_in)

        # create input for controlling noise output if that's what we decide
        # that we want to do
        if self.use_noise:
            z = Input((self.num_hypotheses, self.noise_dim))
            ins += [z]

        y = OneHot(self.num_options)(next_option_in)
        y = Flatten()(y)
        y2 = OneHot(self.num_options)(next_option_in2)
        y2 = Flatten()(y2)
        x = h
        tform = self._makeTransform()
        x = tform([h0,h,y])
        x2 = tform([h0,x,y2])
        image_out = decoder([x])
        image_out2 = decoder([x2])

        self.transform_model = tform

        # =====================================================================
        # Make the discriminator
        image_discriminator = self._makeImageDiscriminator(img_shape)
        self.discriminator = image_discriminator

        image_discriminator.trainable = False
        is_fake = image_discriminator([
            img0_in, img_in,
            next_option_in, 
            next_option2_in,
            image_out,
            image_out2])

        # =====================================================================
        actor = GetHuskyActorModel(h, self.num_options, pose_size,
                self.decoder_dropout_rate)
        actor.compile(loss="mae",optimizer=self.getOptimizer())
        cmd = actor([h, y])
        lfn = self.loss
        lfn2 = "logcosh"
        val_loss = "binary_crossentropy"

        # =====================================================================
        # Create models to train
        predictor = Model(ins + [label_in],
                [image_out, image_out2, next_option_out, value_out])
        predictor.compile(
                loss=[lfn, lfn, "binary_crossentropy", val_loss],
                loss_weights=[1., 1., 0.1, 0.1,],
                optimizer=self.getOptimizer())
        if self.do_all:
            train_predictor = Model(ins + [label_in],
                    [image_out, image_out2, next_option_out, value_out,
                        cmd])
            train_predictor.compile(
                    loss=[lfn, lfn, "binary_crossentropy", val_loss,
                        lfn2,],
                    loss_weights=[1., 1., 0.1, 0.1, 1.,],
                    optimizer=self.getOptimizer())
        else:
            train_predictor = Model(ins + [label_in],
                    [image_out, image_out2,
                        ])
            train_predictor.compile(
                    loss=lfn, 
                    optimizer=self.getOptimizer())
        self.predictor = predictor
        self.train_predictor = train_predictor
        self.actor = actor
