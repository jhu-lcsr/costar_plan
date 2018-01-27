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


class ConditionalSampler2(PredictionSampler2):
    '''
    Version of the sampler that only produces results conditioned on a
    particular action; this version does not bother trying to learn a separate
    distribution for each possible state.
    '''

    def __init__(self, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.

        Parameters:
        -----------
        taskdef: definition of the problem used to create a task model
        '''
        super(ConditionalSampler2, self).__init__(*args, **kwargs)
        self.PredictorCb = ImageCb

    def _makePredictor(self, features):
        # =====================================================================
        # Create many different image decoders
        image_outs = []
        arm_outs = []
        gripper_outs = []
        train_outs = []
        label_outs = []
        
        (images, arm, gripper) = features
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                images,
                arm,
                gripper)

        # =====================================================================
        # Load the image decoders
        img_in = Input(img_shape,name="predictor_img_in")
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        arm_gripper = Concatenate()([arm_in, gripper_in])
        label_in = Input((1,))
        ins = [img_in, arm_in, gripper_in, label_in]

        if self.skip_connections:
            encoder = self._makeImageEncoder2(img_shape)
            decoder = self._makeImageDecoder2(self.hidden_shape)
        else:
            encoder = self._makeImageEncoder(img_shape)
            decoder = self._makeImageDecoder(self.hidden_shape)

        LoadEncoderWeights(self, encoder, decoder)
        image_discriminator = LoadGoalClassifierWeights(self,
                make_classifier_fn=MakeImageClassifier,
                img_shape=img_shape)

        # =====================================================================
        # Load the arm and gripper representation
        rep_channels = self.encoder_channels
        sencoder = self._makeStateEncoder(arm_size, gripper_size, False)
        sdecoder = self._makeStateDecoder(arm_size, gripper_size,
                rep_channels)

        # =====================================================================
        # combine these models together with state information and label
        # information
        hidden_encoder = self._makeToHidden(img_shape, arm_size, gripper_size,
                rep_channels)
        hidden_decoder = self._makeFromHidden(rep_channels)

        try:
            hidden_encoder.load_weights(self.makeName(
                "pretrain_sampler",
                "hidden_encoder"))
            hidden_decoder.load_weights(self.makeName(
                "pretrain_sampler",
                "hidden_decoder"))
            hidden_encoder.trainable = self.retrain
            hidden_decoder.trainable = self.retrain
        except Exception as e:
            pass

        h = hidden_encoder(ins)
        value_out, next_option_out = GetNextOptionAndValue(h,
                                                           self.num_options,
                                                           self.rep_size,
                                                           dropout_rate=0.5,
                                                           option_in=None)

        # create input for controlling noise output if that's what we decide
        # that we want to do
        if self.use_noise:
            z = Input((self.num_hypotheses, self.noise_dim))
            ins += [z]

        next_option_in = Input((1,), name="next_option_in")
        ins += [next_option_in]

        y = OneHot(self.num_options)(next_option_in)
        y = Flatten()(y)
        x = h
        x = AddConv2D(x, self.tform_filters*2, [1,1], 1, 0.)
        for i in range(self.num_transforms):
            print(x,y)
            x = TileOnto(x, y, self.num_options, (8,8))
            x = AddConv2D(x, self.tform_filters*2,
                    self.tform_kernel_size,
                    stride=1,
                    dropout_rate=self.tform_dropout_rate)
        x =  Concatenate(axis=-1)([x,h])
        x = AddConv2D(x, rep_channels, [1, 1], stride=1,
                dropout_rate=0.)
        image_out, arm_out, gripper_out, label_out = hidden_decoder(x)

        # =====================================================================
        # Create models to train
        predictor = Model(ins,
                [image_out, arm_out, gripper_out, label_out, next_option_out,
                    value_out])
        actor = None
        predictor.compile(
                loss=["mae", "logcosh", "logcosh", "categorical_crossentropy", "categorical_crossentropy",
                      "binary_crossentropy"],
                loss_weights=[1., 1., 0.2, 0.025, 0.1, 0.],
                optimizer=self.getOptimizer())
        return predictor, predictor, actor, ins, h

    def _getData(self, *args, **kwargs):
        features, targets = GetAllMultiData(self.num_options, *args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
        if self.use_noise:
            noise_len = features[0].shape[0]
            z = np.random.random(size=(noise_len,self.num_hypotheses,self.noise_dim))
            return [I, q, g, oin, z, o1], [o1_1h, v, I_target, q_target, g_target,
                    o1_1h,
                    o1_1h]
        else:
            return [I, q, g, oin, o1], [I_target, q_target, g_target, o1_1h,
                    o1_1h,
                    v,]

