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
            decoder = self._makeImageDecoder2(self.hidden_shape)
        else:
            encoder = self._makeImageEncoder(img_shape)
            decoder = self._makeImageDecoder(self.hidden_shape)

        LoadEncoderWeights(self, encoder, decoder, gan=False)
        image_discriminator = LoadGoalClassifierWeights(self,
                make_classifier_fn=MakeImageClassifier,
                img_shape=img_shape)

        # =====================================================================
        # Load the arm and gripper representation
        if self.skip_connections:
            h, s32, s16, s8 = encoder([img0_in, img_in])
        else:
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
        disc_out2 = image_discriminator([img0_in, image_out2])

        # =====================================================================
        # Create models to train
        model = Model(ins + [label_in],
                [image_out, image_out2, disc_out2])
        if self.no_disc:
            disc_wt = 0.
        else:
            disc_wt = 1e-3
        model.compile(
                loss=[self.loss, self.loss, "categorical_crossentropy"],
                loss_weights=[1., 1., disc_wt],
                optimizer=self.getOptimizer())
        self.model = model

    def _getData(self, *args, **kwargs):
        return GetConditionalHuskyData(self.num_options, *args, **kwargs)
