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

        encoder = self._makeImageEncoder(img_shape)
        decoder = self._makeImageDecoder(self.hidden_shape)

        LoadEncoderWeights(self, encoder, decoder, gan=False)

        # =====================================================================
        # Load the arm and gripper representation
        h = encoder([img_in])
        h0 = encoder(img0_in)

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

        if self.validate:
            self.loadValidationModels(pose_size, h0, h)

        if not self.no_disc:
            image_discriminator = LoadGoalClassifierWeights(self,
                    make_classifier_fn=MakeImageClassifier,
                    img_shape=img_shape)
            disc_out2 = image_discriminator([img0_in, image_out2])

        # =====================================================================
        # Create models to train
        if self.no_disc:
            disc_wt = 0.
        else:
            disc_wt = 1e-3
        if self.no_disc:
            model = Model(ins + [label_in],
                    [image_out, image_out2,])
            model.compile(
                    loss=[self.loss, self.loss,],
                    loss_weights=[1., 1.,],
                    optimizer=self.getOptimizer())
        else:
            model = Model(ins + [label_in],
                    [image_out, image_out2, disc_out2])
            model.compile(
                    loss=[self.loss, self.loss, "categorical_crossentropy"],
                    loss_weights=[1., 1., disc_wt],
                    optimizer=self.getOptimizer())
        self.model = model

    def loadValidationModels(self, pose_size, h0, h):

        pose_in = Input((pose_size,))
        label_in = Input((1,))

        print(">>> GOAL_CLASSIFIER")
        image_discriminator = LoadGoalClassifierWeights(self,
                    make_classifier_fn=MakeImageClassifier,
                    img_shape=(64, 64, 3))
        image_discriminator.compile(loss="categorical_crossentropy",
                                    metrics=["accuracy"],
                                    optimizer=self.getOptimizer())
        self.discriminator = image_discriminator

        print(">>> VALUE MODEL")
        self.value_model = GetValueModel(h, self.num_options, 128,
                self.decoder_dropout_rate)
        self.value_model.compile(loss="mae", optimizer=self.getOptimizer())
        self.value_model.load_weights(self.makeName("secondary", "value"))

        print(">>> NEXT MODEL")
        self.next_model = GetNextModel(h, self.num_options, 128,
                self.decoder_dropout_rate)
        self.next_model.compile(loss="mae", optimizer=self.getOptimizer())
        self.next_model.load_weights(self.makeName("secondary", "next"))

        print(">>> ACTOR MODEL")
        self.actor = GetHuskyActorModel(h, self.num_options, pose_size,
                self.decoder_dropout_rate)
        self.actor.compile(loss="mae",optimizer=self.getOptimizer())
        self.actor.load_weights(self.makeName("secondary", "actor"))

        print(">>> POSE MODEL")
        self.pose_model = GetHuskyPoseModel(h, self.num_options, pose_size,
                self.decoder_dropout_rate)
        self.pose_model.compile(loss="mae",optimizer=self.getOptimizer())
        self.pose_model.load_weights(self.makeName("secondary", "pose"))

        print(">>> Q MODEL")
        self.q_model = GetNextModel(h, self.num_options, 128,
                self.decoder_dropout_rate)
        self.q_model.compile(loss="mae", optimizer=self.getOptimizer())
        self.q_model.load_weights(self.makeName("secondary", "q"))



    def _getData(self, *args, **kwargs):
        return GetConditionalHuskyData(self.validate, self.no_disc, self.num_options, *args, **kwargs)
