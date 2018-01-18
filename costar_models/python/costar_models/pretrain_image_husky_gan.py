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
from .husky_sampler import *
from .pretrain_image_gan import *

class PretrainImageHuskyGan(PretrainImageGan):

    def _makeModel(self, image, pose, *args, **kwargs):
        '''
        Little helper function wraps makePredictor to consturct all the models.

        Parameters:
        -----------
        image, arm, gripper: variables of the appropriate sizes
        '''
        self.predictor, self.train_predictor, self.actor, ins, hidden = \
            self._makePredictor(
                (image, pose))
        if self.train_predictor is None:
            raise RuntimeError('did not make trainable model')


    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, pose) = features
        img_shape = images.shape[1:]
        pose_size = pose.shape[-1]

        img_in = Input(img_shape,name="predictor_img_in")
        test_in = Input(img_shape, name="descriminator_test_in")

        encoder = self._makeImageEncoder(img_shape)
        enc = encoder([img_in])
        decoder = self._makeImageDecoder(
                self.hidden_shape,
                self.skip_shape, False)

        if self.load_pretrained_weights:
            try:
                encoder.load_weights(self._makeName(
                    "pretrain_image_encoder_model_husky",
                    "image_encoder.h5f"))
                decoder.load_weights(self._makeName(
                    "pretrain_image_encoder_model_husky",
                    "image_decoder.h5f"))
            except Exception as e:
                print(">> Failed to load pretrained generator weights.")

        gen_out = decoder(enc)
        image_discriminator = self._makeImageDiscriminator(img_shape)
        self.discriminator = image_discriminator

        image_discriminator.trainable = False
        o1 = image_discriminator([img_in, gen_out])

        self.model = Model([img_in], [gen_out, o1])
        self.model.compile(
                loss=["mae"] + ["binary_crossentropy"],
                loss_weights=[100., 1.],
                optimizer=self.getOptimizer())

        self.generator = Model([img_in], [gen_out])
        self.generator.compile(
                loss=["logcosh"],
                optimizer=self.getOptimizer())

        image_discriminator.summary()

        return self.model, self.model, None, [img_in], enc

    def _getData(self, image, *args, **kwargs):
        I = np.array(image) / 255.
        return [I], [I]

