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

from .robot_multi_models import *
from .mhp_loss import *
from .loss import *
from .sampler2 import *

from .conditional_image import ConditionalImage
from .dvrk import *

class ConditionalImageJigsaws(ConditionalImage):

    def __init__(self, *args, **kwargs):

        super(ConditionalImageJigsaws, self).__init__(*args, **kwargs)

        self.num_options = SuturingNumOptions()
        self.PredictorCb = PredictorShowImageOnlyMultiStep

    def _makeModel(self, image, *args, **kwargs):

        img_shape = image.shape[1:]
        img_size = 1.
        for dim in img_shape:
            img_size *= dim

        img0_in = Input(img_shape, name="predictor_img0_in")
        img_in = Input(img_shape, name="predictor_img_in")
        prev_option_in = Input((1,), name="predictor_prev_option_in")
        ins = [img0_in, img_in]

        if self.skip_connections:
            encoder = self._makeImageEncoder2(img_shape)
            decoder = self._makeImageDecoder2(self.hidden_shape)
        else:
            encoder = MakeJigsawsImageEncoder(self, img_shape)
            decoder = MakeJigsawsImageDecoder(self, self.hidden_shape)

        # =====================================================================
        # Load weights and stuff
        LoadEncoderWeights(self, encoder, decoder, gan=False)
        image_discriminator = LoadGoalClassifierWeights(self,
                make_classifier_fn=MakeJigsawsImageClassifier,
                img_shape=img_shape)

        # =====================================================================
        # Create encoded state
        if self.skip_connections:
            h, s32, s16, s8 = encoder([img0_in, img_in])
        else:
            h = encoder(img_in)
            h0 = encoder(img0_in)

        # Create model for predicting label
        #next_model = GetJigsawsNextModel(h, self.num_options, 128,
        #        self.decoder_dropout_rate)
        #next_model.compile(loss="mae", optimizer=self.getOptimizer())
        #next_option_out = next_model([h0, h, prev_option_in])
        #self.next_model = next_model

        option_in = Input((1,), name="option_in")
        option_in2 = Input((1,), name="option_in2")
        ins += [option_in, option_in2]

        # --------------------------------------------------------------------
        # Image model
        h_dim = (12, 16)

        y = Flatten()(OneHot(self.num_options)(option_in))
        y2 = Flatten()(OneHot(self.num_options)(option_in2))
        x = h
        tform = MakeJigsawsTransform(self, h_dim=(12,16), small=True)
        l = [h0, h, y, z1] if self.use_noise else [h0, h, y]
        x = tform(l)
        l = [h0, x, y2, z2] if self.use_noise else [h0, x, y]
        x2 = tform(l)
        image_out, image_out2 = decoder([x]), decoder([x2])
        disc_out2 = image_discriminator([image_out2])

        # =====================================================================
        # Create models to train
        model = Model(ins + [prev_option_in],
                [image_out, image_out2, disc_out2])
        # --------------------------------------------------------------------
        # Create multiple hypothesis loss
        if self.no_disc:
            disc_wt = 0.
        else:
            disc_wt = 1e-3
        model.compile(
                loss=[self.loss, self.loss, "categorical_crossentropy"],
                loss_weights=[1., 1., disc_wt],
                optimizer=self.getOptimizer())

        self.model = model
        self.model.summary()

    def _getData(self, image, label, goal_image, goal_label,
            prev_label, *args, **kwargs):

        image = np.array(image) / 255.
        goal_image = np.array(goal_image) / 255.

        goal_image2, label2 = GetNextGoal(goal_image, label)

        # Extend image_0 to full length of sequence
        image0 = image[0,:,:,:]
        length = image.shape[0]
        image0 = np.tile(np.expand_dims(image0,axis=0),[length,1,1,1])

        label_1h = np.squeeze(ToOneHot2D(label, self.num_options))
        label2_1h = np.squeeze(ToOneHot2D(label2, self.num_options))
        return ([image0, image, label, goal_label, prev_label],
                [np.expand_dims(goal_image, axis=1),
                 #np.expand_dims(goal_image2, axis=1), label_1h])#, label2_1h]
                 np.expand_dims(goal_image2, axis=1),
                 np.expand_dims(label2_1h, axis=1)])

