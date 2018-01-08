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


class ConditionalImage(PredictionSampler2):
    '''
    Version of the sampler that only produces results conditioned on a
    particular action; this version does not bother trying to learn a separate
    distribution for each possible state.

    This one generates:
      - image
      - arm command
      - gripper command
    '''

    def __init__(self, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.

        Parameters:
        -----------
        taskdef: definition of the problem used to create a task model
        '''
        super(ConditionalImage, self).__init__(*args, **kwargs)
        self.PredictorCb = ImageWithFirstCb
        self.rep_size = 256
        self.num_transforms = 3
        self.do_all = True
        self.transform_model = None
        self.skip_connections = False

    def _makeTransform(self):
        h = Input((8,8,self.encoder_channels),name="h_in")
        #h0 = Input((8,8,self.encoder_channels),name="h0_in")
        option = Input((48,),name="t_opt_in")
        x, y = h, option
        #x = Concatenate()([h, h0])
        x0 = AddConv2D(x, self.tform_filters*2, [1,1], 1, 0.)
        x = x0
        for i in range(self.num_transforms):
            x = TileOnto(x, y, self.num_options, (8,8))
            x = AddConv2D(x, self.tform_filters*2,
                    self.tform_kernel_size,
                    stride=1,
                    dropout_rate=self.tform_dropout_rate)
        x =  Concatenate(axis=-1)([x,x0])
        x = AddConv2D(x, self.encoder_channels, [1, 1], stride=1,
                dropout_rate=0.)

        #self.transform_model = Model([h0,h,option], x, name="tform")
        self.transform_model = Model([h,option], x, name="tform")
        self.transform_model.compile(loss="mae", optimizer=self.getOptimizer())
        self.transform_model.summary()
        return self.transform_model

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
        img0_in = Input(img_shape,name="predictor_img0_in")
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        arm_gripper = Concatenate()([arm_in, gripper_in])
        label_in = Input((1,))
        ins = [img0_in, img_in]

        if self.skip_connections:
            encoder = self._makeImageEncoder2(img_shape)
        else:
            encoder = self._makeImageEncoder(img_shape)
            encoder0 = self._makeImageEncoder(img_shape, copy=True)
        try:
            encoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
            encoder0.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                "image_encoder.h5f"))
            encoder0.trainable = True
        except Exception as e:
            if not self.retrain:
                raise e

        if self.skip_connections:
            decoder = self._makeImageDecoder2(self.hidden_shape)
        else:
            decoder = self._makeImageDecoder(self.hidden_shape)
        try:
            decoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
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
            h = encoder([img_in, img0_in])
            #h0 = encoder(img0_in)

        next_model = GetNextModel(h, self.num_options, 128,
                self.decoder_dropout_rate)
        value_model = GetValueModel(h, self.num_options, 64,
                self.decoder_dropout_rate)
        next_model.compile(loss="mae", optimizer=self.getOptimizer())
        value_model.compile(loss="mae", optimizer=self.getOptimizer())
        #value_out = value_model([h0,h,label_in])
        #next_option_out = next_model([h0,h,label_in])
        value_out = value_model([h,label_in])
        next_option_out = next_model([h,label_in])
        self.next_model = next_model
        self.value_model = value_model

        # create input for controlling noise output if that's what we decide
        # that we want to do
        if self.use_noise:
            z = Input((self.num_hypotheses, self.noise_dim))
            ins += [z]

        next_option_in = Input((48,), name="next_option_in")
        ins += [next_option_in]

        #y = OneHot(self.num_options)(next_option_in)
        #y = Flatten()(y)
        y = next_option_in
        x = h
        tform = self._makeTransform()
        #x = tform([h0,h,y])
        x = tform([h,y])
        image_out = decoder([x, s32, s16, s8])

        # =====================================================================
        actor = GetActorModel(h, self.num_options, arm_size, gripper_size,
                self.decoder_dropout_rate)
        actor.compile(loss="mae",optimizer=self.getOptimizer())
        arm_cmd, gripper_cmd = actor([h, next_option_in])
        lfn = self.loss
        lfn2 = "logcosh"
        val_loss = "binary_crossentropy"

        # =====================================================================
        # Create models to train
        predictor = Model(ins + [label_in],
                [image_out, next_option_out, value_out])
        predictor.compile(
                loss=[lfn, "binary_crossentropy", val_loss],
                loss_weights=[1., 0.1, 0.1,],
                optimizer=self.getOptimizer())
        if self.do_all:
            train_predictor = Model(ins + [label_in],
                    [image_out, next_option_out, value_out, #o1, o2,
                        arm_cmd,
                        gripper_cmd])
            train_predictor.compile(
                    loss=[lfn, "binary_crossentropy", val_loss,
                        lfn2, lfn2],
                    loss_weights=[1., 0.1, 0.1, 1., 0.2],
                    optimizer=self.getOptimizer())
        else:
            train_predictor = Model(ins + [label_in],
                    [image_out, #o1, o2,
                        ])
            train_predictor.compile(
                    loss=[lfn], 
                    optimizer=self.getOptimizer())
        actor.summary()
        train_predictor.summary()
        return predictor, train_predictor, actor, ins, h

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        oin_1h = np.squeeze(self.toOneHot2D(oin, self.num_options))
        qa = np.squeeze(qa)
        ga = np.squeeze(ga)
        if self.do_all:
            if self.use_noise:
                noise_len = features[0].shape[0]
                z = np.random.random(size=(noise_len,self.num_hypotheses,self.noise_dim))
                return [I0, I, z, o1, oin], [ I_target, o1, v, qa, ga]
            else:
                return [I0, I, o1, oin], [ I_target, o1, v, qa, ga]
        else:
            if self.use_noise:
                noise_len = features[0].shape[0]
                z = np.random.random(size=(noise_len,self.num_hypotheses,self.noise_dim))
                return [I0, I, z, o1, oin], [ I_target]
            else:
                return [I0, I, o1, oin], [ I_target]


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

        p = self.next_model.predict([h0, hidden, prev_option])
        #p = np.exp(p)
        #p /= np.sum(p)
        return p

    def value(self, hidden, prev_option, features):
        h0 = self.encodeInitial(features)
        v = self.value_model.predict([h0, hidden, prev_option])
        return v

    def transform(self, hidden, option_in=-1):

        raise NotImplementedError('transform() not implemented')

    def act(self, *args, **kwargs):
        raise NotImplementedError('act() not implemented')

    def debugImage(self, features):
        return features[1]
