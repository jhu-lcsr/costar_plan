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

class PretrainImageHuskyGan(PretrainImageGan):

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

        if self.load_pretrained_weights:
            try:
                encoder.load_weights(self._makeName(
                    "pretrain_image_encoder_model",
                    "image_encoder.h5f"))
                decoder.load_weights(self._makeName(
                    "pretrain_image_encoder_model",
                    "image_decoder.h5f"))
            except Exception as e:
                if not self.retrain:
                    raise e

        gen_out = decoder(enc)
        image_discriminator = self._makeImageDiscriminator(img_shape)
        self.discriminator = image_discriminator

        image_discriminator.trainable = False
        o1 = image_discriminator([img_in, gen_out])

        encoder.summary()
        decoder.summary()
        image_discriminator.summary()

        self.model = Model([img_in], [gen_out, o1])
        self.model.compile(
                loss=["mae"] + ["binary_crossentropy"],
                loss_weights=[100., 1.],
                optimizer=self.getOptimizer())
        self.model.summary()

        self.generator = Model([img_in], [gen_out])
        self.generator.compile(
                loss=["logcosh"],
                optimizer=self.getOptimizer())
        self.generator.summary()

        return self.model, self.model, None, [img_in], enc

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, oin, q_target] = features
        o1 = targets[1]
        oin_1h = np.squeeze(self.toOneHot2D(oin, self.num_options))
        return [I], [I, oin_1h, oin_1h]

