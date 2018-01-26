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
from .pretrain_image_gan import wasserstein_loss

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

        # Create inputs
        img0_in = Input(img_shape, name="predictor_img0_in")
        img_in = Input(img_shape, name="predictor_img_in")
        ins = [img0_in, img_in]

        # next option - used to compute the next image 
        option_in = Input((1,), name="option_in")
        option_in2 = Input((1,), name="option_in2")
        ins += [option_in, option_in2]

        # =====================================================================
        # Load weights and stuff. We'll load the GAN version of the weights.
        encoder = MakeJigsawsImageEncoder(self, img_shape)
        decoder = MakeJigsawsImageDecoder(self, self.hidden_shape)
        LoadEncoderWeights(self, encoder, decoder, gan=True)

        # =====================================================================
        # Create outputs
        if self.skip_connections:
            h, s32, s16, s8 = encoder([img0_in, img_in])
        else:
            h = encoder(img_in)
            h0 = encoder(img0_in)

        y = Flatten()(OneHot(self.num_options)(option_in))
        y2 = Flatten()(OneHot(self.num_options)(option_in2))
        x = h
        tform = MakeJigsawsTransform(self, h_dim=(12,16), small=True)
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
        loss = wasserstein_loss if self.use_wasserstein else "binary_crossentropy"
        model.compile(
                loss=["mae", "mae", loss],
                loss_weights=[100., 100., 1.],
                optimizer=self.getOptimizer())
        self.discriminator.summary()
        model.summary()
        self.model = model

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

        x0 = AddConv2D(img0, 32, [4,4], 1, dr, "same", lrelu=True, bn=False)
        xobs = AddConv2D(img, 32, [4,4], 1, dr, "same", lrelu=True, bn=False)
        xg1 = AddConv2D(img_goal, 32, [4,4], 1, dr, "same", lrelu=True, bn=False)
        xg2 = AddConv2D(img_goal2, 32, [4,4], 1, dr, "same", lrelu=True, bn=False)

        #x1 = Add()([x0, xobs, xg1])
        #x2 = Add()([x0, xg1, xg2])
        x1 = Add()([xobs, xg1])
        x2 = Add()([xg1, xg2])
        
        # -------------------------------------------------------------
        y = OneHot(self.num_options)(option2)
        y = AddDense(y, 32, "lrelu", dr)
        x2 = TileOnto(x2, y, 32, img_size, add=True)
        x2 = AddConv2D(x2, 32, [4,4], 2, dr, "valid", lrelu=True, bn=False)

        # Final block
        x2 = AddConv2D(x2, 64, [4,4], 2, dr, "valid", lrelu=True, bn=True)
        x2 = AddConv2D(x2, 128, [4,4], 2, dr, "valid", lrelu=True, bn=True)
        x2 = AddConv2D(x2, 256, [4,4], 2, dr, "valid", lrelu=True, bn=True)
        #x = Concatenate(axis=-1)([x1, x2])
        #x = Add()([x1, x2])
        #x = AddConv2D(x2, 1, [1,1], 1, 0., "same", l, bn=True)

        # Combine
        #x = AveragePooling2D(pool_size=(12,16))(x)
        #x = AveragePooling2D(pool_size=(24,32))(x)
        x = x2
        x = Flatten()(x)
        x = AddDense(x, 1, "linear", 0., output=True, bn=False)
        discrim = Model(ins, x, name="image_discriminator")
        self.lr *= 2.
        loss = wasserstein_loss if self.use_wasserstein else "binary_crossentropy"
        discrim.compile(loss=loss, optimizer=self.getOptimizer())
        self.lr *= 0.5
        self.image_discriminator = discrim
        return discrim


