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
            encoder.summary()
            encoder.load_weights(self._makeName(
                #pretrain_image_encoder_model",
                "pretrain_image_gan_model",
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
                #"pretrain_image_encoder_model",
                "pretrain_image_gan_model",
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

        if self.skip_connections:
            h, s32, s16, s8 = encoder([img0_in, img_in])
        else:
            #h = encoder([img_in, img0_in])
            h = encoder([img_in])
            h0 = encoder(img0_in)
        next_option_in = Input((self.num_options,), name="next_option_in")
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
        is_fake = image_discriminator([img0_in, img_in, next_option_in, image_out])

        # =====================================================================
        # Create generator model to train
        lfn = self.loss
        predictor = Model(ins,
                [image_out])
        predictor.compile(
                loss=[lfn],
                optimizer=self.getOptimizer())
        self.generator = predictor

        # =====================================================================
        # And adversarial model 
        model = Model(ins, [image_out, is_fake])
        model.compile(
                loss=["mae"] + ["binary_crossentropy"],
                loss_weights=[100., 1.],
                optimizer=self.getOptimizer())
        model.summary()
        self.model = model

        return predictor, model, model, ins, h

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

    def _makeImageDiscriminator(self, img_shape):
        '''
        create image-only encoder to extract keypoints from the scene.
        
        Params:
        -------
        img_shape: shape of the image to encode
        '''
        img0 = Input(img_shape,name="img0_encoder_in")
        img = Input(img_shape,name="img_encoder_in")
        img_goal = Input(img_shape,name="goal_encoder_in")
        option = Input((self.num_options,),name="disc_options")
        ins = [img0, img, option, img_goal]
        dr = self.dropout_rate
        dr = 0
        x = AddConv2D(img, 64, [4,4], 1, dr, "valid", lrelu=True)
        x0 = AddConv2D(img0, 64, [4,4], 1, dr, "valid", lrelu=True)
        xg = AddConv2D(img_goal, 64, [4,4], 1, dr, "valid", lrelu=True)
        x = Add()([x, x0])
        x = AddConv2D(x, 64, [4,4], 2, dr, "valid", lrelu=True)


        #x = TileOnto(x, y, 64, (29,29))
        x = AddConv2D(x, 64, [4,4], 1, dr, "same", lrelu=True)

        x = AddConv2D(x, 128, [4,4], 2, dr, "valid", lrelu=True)
        x = AddConv2D(x, 128, [4,4], 1, dr, "same", lrelu=True)

        x = AddConv2D(x, 256, [4,4], 2, dr, "valid", lrelu=True)

        y = AddDense(option, 256, "lrelu", dr)
        print (x,y)
        x = TileOnto(x, y, 64, (29,29))

        x = AddConv2D(x, 256, [4,4], 1, dr, "same", lrelu=True)
        x = AddConv2D(x, 1, [4,4], 1, 0., "valid", activation="sigmoid")
        #x = MaxPooling2D(pool_size=(8,8))(x)
        print("out=",x)
        x = AveragePooling2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        discrim = Model(ins, x, name="image_discriminator")
        discrim.compile(loss="binary_crossentropy", loss_weights=[1.],
                optimizer=self.getOptimizer())
        self.image_discriminator = discrim
        return discrim


