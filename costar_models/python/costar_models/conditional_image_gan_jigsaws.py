from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Input, RepeatVector, Reshape
from keras.layers.merge import Concatenate, Multiply
from keras.models import Model, Sequential
from matplotlib import pyplot as plt

from .conditional_image_gan import ConditionalImageGan
from .dvrk import *
from .data_utils import *

class ConditionalImageGanJigsaws(ConditionalImageGan):
    '''
    Version of the sampler that only produces results conditioned on a
    particular action; this version does not bother trying to learn a separate
    distribution for each possible state.
    '''

    def __init__(self, *args, **kwargs):

        super(ConditionalImageGanJigsaws, self).__init__(*args, **kwargs)

        self.num_options = 16
 
    def _makeModel(self, image, *args, **kwargs):

        img_shape = image.shape[1:]

        img0_in = Input(img_shape, name="predictor_img0_in")
        img_in = Input(img_shape, name="predictor_img_in")
        ins = [img0_in, img_in]

        encoder = MakeJigsawsImageEncoder(self, img_shape)
        decoder = MakeJigsawsImageDecoder(self, self.hidden_shape)

        try:
            encoder.load_weights(self._makeName(
                #pretrain_image_encoder_model",
                "pretrain_image_gan_model_jigsaws",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
            decoder.load_weights(self._makeName(
                #"pretrain_image_encoder_model",
                "pretrain_image_gan_model_jigsaws",
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

        # create input for controlling noise output if that's what we decide
        # that we want to do
        if self.use_noise:
            ins += [Input((self.num_hypotheses, self.noise_dim))]

        # next option - used to compute the next image 
        option_in = Input((1,), name="option_in")
        option_in2 = Input((1,), name="option_in2")
        ins += [option_in, option_in2]

        y = Flatten()(OneHot(self.num_options)(option_in))
        y2 = Flatten()(OneHot(self.num_options)(option_in2))
        x = h
        tform = self._makeTransform(h_dim=(12,16))
        x = tform([h0, h, y])
        x2 = tform([h0, x, y2])
        image_out, image_out2 = decoder([x]), decoder([x2])

        self.transform_model = tform

        # =====================================================================
        # Make the discriminator
        image_discriminator = self._makeImageDiscriminator(img_shape)
        self.discriminator = image_discriminator

        image_discriminator.trainable = False
        is_fake = image_discriminator([
            img0_in, img_in,
            option_in, option_in2,
            image_out, image_out2])

        # =====================================================================
        # Create generator model to train
        lfn = self.loss
        generator = Model(ins, [image_out, image_out2])
        generator.compile(
                loss=[lfn, lfn],
                optimizer=self.getOptimizer())
        self.generator = generator

        # =====================================================================
        # And adversarial model 
        model = Model(ins, [image_out, image_out2, is_fake])
        model.compile(
                loss=["mae", "mae", "binary_crossentropy"],
                loss_weights=[100., 100., 1.],
                optimizer=self.getOptimizer())
        model.summary()
        self.discriminator.summary()
        self.model = model
        self.train_predictor = model

        self.predictor = generator

    def _getData(self, image, label, goal_image, goal_label,
            prev_label, *args, **kwargs):

        image = np.array(image) / 255.
        goal_image = np.array(goal_image) / 255.

        goal_image2, _ = GetNextGoal(goal_image, label)

        # Extend image_0 to full length of sequence
        image0 = image[0,:,:,:]
        length = image.shape[0]
        image0 = np.tile(np.expand_dims(image0,axis=0),[length,1,1,1])
        return [image0, image, label, goal_label], [goal_image, goal_image2]

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
        img_goal2 = Input(img_shape,name="goal2_encoder_in")
        option = Input((1,),name="disc_options")
        option2 = Input((1,),name="disc2_options")
        ins = [img0, img, option, option2, img_goal, img_goal2]
        dr = self.dropout_rate
        dr = 0
        img_size = (96, 128)

        x0 = AddConv2D(img0, 64, [4,4], 1, dr, "same", lrelu=True, bn=False)
        xobs = AddConv2D(img, 64, [4,4], 1, dr, "same", lrelu=True, bn=False)
        xg1 = AddConv2D(img_goal, 64, [4,4], 1, dr, "same", lrelu=True, bn=False)
        xg2 = AddConv2D(img_goal2, 64, [4,4], 1, dr, "same", lrelu=True, bn=False)

        x1 = Add()([x0, xobs, xg1])
        x2 = Add()([x0, xg1, xg2])
        
        # -------------------------------------------------------------
        y = OneHot(self.num_options)(option)
        y = AddDense(y, 64, "lrelu", dr)
        x1 = TileOnto(x1, y, 64, img_size, add=True)
        x1 = AddConv2D(x1, 64, [4,4], 2, dr, "same", lrelu=True, bn=False)

        # -------------------------------------------------------------
        y = OneHot(self.num_options)(option2)
        y = AddDense(y, 64, "lrelu", dr)
        x2 = TileOnto(x2, y, 64, img_size, add=True)
        x2 = AddConv2D(x2, 64, [4,4], 2, dr, "same", lrelu=True, bn=False)
        x = x2
        #x = Concatenate()([x1, x2])
        x = AddConv2D(x, 128, [4,4], 2, dr, "same", lrelu=True)
        #x = AddConv2D(x, 128, [4,4], 1, dr, "same", lrelu=True)
        x= AddConv2D(x, 256, [4,4], 2, dr, "same", lrelu=True)
        #x= AddConv2D(x, 512, [4,4], 2, dr, "same", lrelu=True)
        #x = AddConv2D(x, 256, [4,4], 1, dr, "same", lrelu=True)
        x = AddConv2D(x, 1, [4,4], 1, 0., "same", activation="sigmoid")

        #x = MaxPooling2D(pool_size=(8,8))(x)
        x = AveragePooling2D(pool_size=(12,16))(x)
        x = Flatten()(x)
        discrim = Model(ins, x, name="image_discriminator")
        self.lr *= 2.
        discrim.compile(loss="binary_crossentropy", loss_weights=[1.],
                optimizer=self.getOptimizer())
        self.lr *= 0.5
        self.image_discriminator = discrim
        return discrim


