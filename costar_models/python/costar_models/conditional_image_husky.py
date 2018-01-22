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

from .conditional_image import *
from .husky import *
from .planner import *

class ConditionalImageHusky(ConditionalImage):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(ConditionalImageHusky, self).__init__(taskdef, *args, **kwargs)
        self.num_options = HuskyNumOptions()
        self.null_option = HuskyNullOption()

    def _makeModel(self, image, pose, *args, **kwargs):
       
        img_shape = image.shape[1:]
        pose_size = pose.shape[-1]

        # =====================================================================
        # Load the image decoders
        img_in = Input(img_shape,name="predictor_img_in")
        img0_in = Input(img_shape,name="predictor_img0_in")
        label_in = Input((1,))
        ins = [img0_in, img_in]

        if self.skip_connections:
            encoder = self._makeImageEncoder2(img_shape)
        else:
            encoder = self._makeImageEncoder(img_shape)
        try:
            encoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                #"pretrain_image_gan_model_husky",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
        except Exception as e:
            if not self.retrain:
                raise e

        if self.skip_connections:
            decoder = self._makeImageDecoder2(self.hidden_shape)
        else:
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

        next_model = GetNextModel(h, self.num_options, 128,
                self.decoder_dropout_rate)
        value_model = GetValueModel(h, self.num_options, 64,
                self.decoder_dropout_rate)
        next_model.compile(loss="mae", optimizer=self.getOptimizer())
        value_model.compile(loss="mae", optimizer=self.getOptimizer())
        value_out = value_model([h])
        next_option_out = next_model([h0,h,label_in])

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
        x = tform([h0,h,y])
        x2 = tform([h0,x,y2])
        image_out = decoder([x])
        image_out2 = decoder([x2])
        #image_out = decoder([x, s32, s16, s8])

        image_discriminator = MakeImageClassifier(self, img_shape)
        image_discriminator.load_weights(
                self._makeName("goal_discriminator_model_husky", "predictor_weights.h5f"))
        image_discriminator.trainable = False
 
        disc_out2 = image_discriminator(image_out2)

        self.next_model = next_model
        self.value_model = value_model
        self.transform_model = tform

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
            model = Model(ins + [label_in],
                    [image_out, image_out2, next_option_out, value_out,
                        cmd])
            model.compile(
                    loss=[lfn, lfn, "binary_crossentropy", val_loss,
                        lfn2,],
                    loss_weights=[1., 1., 0.1, 0.1, 1.,],
                    optimizer=self.getOptimizer())
        else:
            model = Model(ins + [label_in],
                    [image_out, image_out2,
                        ])
            model.compile(
                    loss=lfn, 
                    optimizer=self.getOptimizer())
        self.predictor = predictor
        self.model = model
        self.actor = actor

    def _getData(self, *args, **kwargs):
        return GetConditionalHuskyData(self.do_all, self.num_options, *args, **kwargs)
