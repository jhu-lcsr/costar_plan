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
from .conditional_image import *
from .pretrain_image_gan import *

class ConditionalImageGan(PretrainImageGan):
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
        super(ConditionalImageGan, self).__init__(*args, **kwargs)
        self.PredictorCb = ImageWithFirstCb
        self.rep_size = 256
        self.num_transforms = 3
        self.do_all = True
        self.transform_model = None
        self.skip_connections = False
 
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
            #encoder0 = self._makeImageEncoder(img_shape, copy=True)
        try:
            encoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                #"pretrain_image_gan_model",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
            #encoder0.load_weights(self._makeName(
            #    "pretrain_image_encoder_model",
            #    "image_encoder.h5f"))
            #encoder0.trainable = self.retrain
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
                #"pretrain_image_gan_model",
                "image_decoder.h5f"))
            decoder.trainable = self.retrain
        except Exception as e:
            if not self.retrain:
                raise e

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
        x = tform([h0,h,y])
        #x = tform([h,y])
        image_out = decoder([x])
        #image_out = decoder([x, s32, s16, s8])

        # =====================================================================
        # Make the discriminator
        image_discriminator = self._makeImageDiscriminator(img_shape)
        self.discriminator = image_discriminator

        image_discriminator.trainable = False
        is_fake = image_discriminator([img_in, gen_out])

        # =====================================================================
        # Create generator model to train
        lfn = self.loss
        predictor = Model(ins + [label_in],
                [image_out])
        predictor.compile(
                loss=[lfn],
                optimizer=self.getOptimizer())

        # =====================================================================
        # And adversarial model 
        model = Model([img_in], [image_out, is_fake])
        model.compile(
                loss=["mae"] + ["binary_crossentropy"],
                loss_weights=[100., 1.],
                optimizer=self.getOptimizer())
        model.summary()

        return predictor, model, None, ins, h

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
                return [I0, I, z, o1], [ I_target]
            else:
                return [I0, I, o1], [ I_target]
        else:
            if self.use_noise:
                noise_len = features[0].shape[0]
                z = np.random.random(size=(noise_len,self.num_hypotheses,self.noise_dim))
                return [I0, I, z, o1], [ I_target]
            else:
                return [I0, I, o1], [ I_target]


