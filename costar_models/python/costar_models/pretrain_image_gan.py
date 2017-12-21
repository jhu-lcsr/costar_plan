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

from .abstract import *
from .callbacks import *
from .robot_multi_models import *
from .split import *
from .mhp_loss import *
from .loss import *
from .multi_sampler import *

class PretrainImageGan(RobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PretrainImageGan, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = ImageCb

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, arm, gripper) = features
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                images,
                arm,
                gripper)

        img_in = Input(img_shape,name="predictor_img_in")
        test_in = Input(img_shape, name="descriminator_test_in")

        encoder = self._makeImageEncoder(img_shape)
        enc = encoder([img_in])
        decoder = self._makeImageDecoder(
                self.hidden_shape,
                self.skip_shape, False)
        out = decoder(enc)

        image_discriminator = self._makeImageEncoder(img_shape, disc=True)
        self.discriminator = image_discriminator

        image_discriminator.trainable = False
        o1 = image_discriminator([img_in, out])

        encoder.summary()
        decoder.summary()
        image_discriminator.summary()

        self.model = Model([img_in], [out, o1])
        self.model.compile(
                loss=["mae"] + ["binary_crossentropy"],
                optimizer=self.getOptimizer())
        self.model.summary()

        self.generator = Model([img_in], [out])
        self.generator.compile(
                loss=["mae"],
                optimizer=self.getOptimizer())
        self.generator.summary()
    
        return self.model, self.model, None, [img_in], enc

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [img, q, g, oin, q_target, g_target,] = features
        return [img], [img]

    def _makeImageEncoder(self, img_shape, disc=False):
        '''
        create image-only decoder to extract keypoints from the scene.
        
        Params:
        -------
        img_shape: shape of the image to encode
        disc: is this being created as part of a discriminator network? If so,
              we handle things slightly differently.
        '''
        img = Input(img_shape,name="img_encoder_in")
        dr = self.dropout_rate
        m = 0.5
        x = img
        x = AddConv2D(x, 32, [5,5], 2, dr, "same", disc, momentum=m)
        if disc:
            img0 = Input(img_shape,name="img0_encoder_in")
            y = img0
            y = AddConv2D(y, 32, [5,5], 2, dr, "same", disc, momentum=m)
            x = Concatenate(axis=-1)([x,y])
            ins = [img0, img]
        else:
            ins = img

        x = AddConv2D(x, 32, [5,5], 1, dr, "same", disc, momentum=m)
        x = AddConv2D(x, 32, [5,5], 1, dr, "same", disc, momentum=m)
        x = AddConv2D(x, 64, [5,5], 2, dr, "same", disc, momentum=m)
        x = AddConv2D(x, 64, [5,5], 1, dr, "same", disc, momentum=m)
        x = AddConv2D(x, 128, [5,5], 2, dr, "same", disc, momentum=m)
        self.encoder_channels = 8
        x = AddConv2D(x, self.encoder_channels, [1,1], 1, 0.*dr,
                "same", disc, momentum=m)

        self.steps_down = 3
        self.hidden_dim = int(img_shape[0]/(2**self.steps_down))
        #self.tform_filters = self.encoder_channels
        self.hidden_shape = (128,)
        x = Flatten()(x)
        #self.hidden_shape = (self.hidden_dim,self.hidden_dim,self.encoder_channels)

        if disc:
            #img0 = Input(img_shape,name="img0_encoder_in")
            #y = img0
            #y = AddConv2D(y, 32, [5,5], 2, dr, "same", disc)
            #y = AddConv2D(y, 32, [5,5], 1, dr, "same", disc)
            #y = AddConv2D(y, 32, [5,5], 1, dr, "same", disc)
            #y = AddConv2D(y, 64, [5,5], 2, dr, "same", disc)
            #y = AddConv2D(y, 64, [5,5], 1, dr, "same", disc)
            #y = AddConv2D(y, 128, [5,5], 2, dr, "same", disc)
            #self.encoder_cyannels = 8
            #y = AddConv2D(y, self.encoder_channels, [1,1], 1, 0.*dr,
            #        "same", disc)
            #x = Concatenate(axis=-1)([x, y])
            x = AddDense(x, 512, "lrelu", dr, output=True)
            x = AddDense(x, 1, "sigmoid", 0, output=True)
            image_encoder = Model(ins, x, name="image_discriminator")
            image_encoder.compile(loss="mae", optimizer=self.getOptimizer())
            self.image_discriminator = image_encoder
        else:
            # dense representation
            x = AddDense(x, self.hidden_shape[0], "relu", dr)
            image_encoder = Model(ins, x, name="image_encoder")
            image_encoder.compile(loss="mae", optimizer=self.getOptimizer())
            self.image_encoder = image_encoder
        return image_encoder

    def _makeImageDecoder(self, hidden_shape, img_shape=None, skip=False):
        '''
        helper function to construct a decoder that will make images.

        parameters:
        -----------
        img_shape: shape of the image, e.g. (64,64,3)
        '''
        rep = Input(hidden_shape,name="decoder_hidden_in")

        x = rep
        if self.hypothesis_dropout:
            dr = self.decoder_dropout_rate
        else:
            dr = 0.
        
        self.steps_up = 3
        hidden_dim = int(img_shape[0]/(2**self.steps_up))
        #self.tform_filters = self.encoder_channels
        (h,w,c) = (hidden_dim,
                    hidden_dim,
                    self.encoder_channels)
        x = AddDense(x, int(h*w*c), "relu", dr)
        x = Reshape((h,w,c))(x)

        #x = AddConv2DTranspose(x, 64, [5,5], 1, dr)
        x = AddConv2DTranspose(x, 128, [1,1], 1, 0.*dr)
        #x = AddConv2DTranspose(x, 64, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 64, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 64, [5,5], 1, dr)
        x = AddConv2DTranspose(x, 32, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 32, [5,5], 1, dr)
        x = AddConv2DTranspose(x, 32, [5,5], 2, dr)
        x = AddConv2DTranspose(x, 32, [5,5], 1, dr)
        ins = rep
        x = Conv2D(3, kernel_size=[1,1], strides=(1,1),name="convert_to_rgb")(x)
        x = Activation("sigmoid")(x)
        decoder = Model(ins, x, name="image_decoder")
        decoder.compile(loss="mae",optimizer=self.getOptimizer())
        self.image_decoder = decoder
        return decoder

    def _fit(self, train_generator, test_generator, callbacks):

        for i in range(self.pretrain_iter):
            # Descriminator pass
            img, _ = train_generator.next()
            img = img[0]
            fake = self.generator.predict(img)
            self.discriminator.trainable = True
            is_fake = np.ones((self.batch_size, 1))
            is_not_fake = np.zeros((self.batch_size, 1))
            res1 = self.discriminator.train_on_batch(
                    [img, img], is_not_fake)
            res2 = self.discriminator.train_on_batch(
                    [img, fake], is_fake)
            self.discriminator.trainable = False
            print("\rPretraining {}/{}: Real loss {}, Fake loss {}".format(
                i, self.pretrain_iter, res1, res2))

        for i in range(self.iter):

            # Descriminator pass
            img, _ = train_generator.next()
            img = img[0]
            fake = self.generator.predict(img)
            self.discriminator.trainable = True
            is_fake = np.ones((self.batch_size, 1))
            is_not_fake = np.zeros((self.batch_size, 1))
            res1 = self.discriminator.train_on_batch(
                    [img, img], is_not_fake)
            res2 = self.discriminator.train_on_batch(
                    [img, fake], is_fake)
            self.discriminator.trainable = False

            # Generator pass
            img, _ = train_generator.next()
            img = img[0]
            res = self.model.train_on_batch(
                    [img], [img, is_not_fake]
            )
            print("\r{}/{}: Gen loss {}, Real loss {}, Fake loss {}".format(
                i, self.iter, res[0], res1, res2))

