from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Input, RepeatVector, Reshape
from keras.layers.merge import Concatenate, Multiply
from keras.losses import binary_crossentropy

from .pretrain_image_gan import *
from .dvrk import *

class PretrainImageJigsawsGan(PretrainImageGan):

    def __init__(self, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PretrainImageJigsawsGan, self).__init__(*args, **kwargs)

        # This is literally the only change from the husky version
        self.num_generator_files = 1

        # Also set up the number of options we expect
        self.num_options = SuturingNumOptions()
        self.save_encoder_decoder = True

    def _makeModel(self, image, *args, **kwargs):
        '''
        Little helper function wraps makePredictor to consturct all the models.

        Parameters:
        -----------
        image, arm, gripper: variables of the appropriate sizes
        '''
        self.predictor, self.train_predictor, self.actor, ins, hidden = \
            self._makePredictor(image)
        if self.train_predictor is None:
            raise RuntimeError('did not make trainable model')

    def _makePredictor(self, images):
        '''
        Create model to predict possible manipulation goals.
        '''
        img_shape = images.shape[1:]

        img_in = Input(img_shape,name="predictor_img_in")
        test_in = Input(img_shape, name="descriminator_test_in")

        encoder = MakeJigsawsImageEncoder(self, img_shape)
        enc = encoder([img_in])
        decoder = MakeJigsawsImageDecoder(
                self,
                self.hidden_shape,
                self.skip_shape, False)

        if self.load_pretrained_weights:
            try:
                encoder.load_weights(self.makeName(
                    "pretrain_image_encoder", "image_encoder"))
                decoder.load_weights(self.makeName(
                    "pretrain_image_encoder", "image_decoder"))
            except Exception as e:
                print(">>> could not load pretrained image weights")
                print(e)

        gen_out = decoder(enc)
        image_discriminator = self._makeImageDiscriminator(img_shape)
        self.discriminator = image_discriminator

        image_discriminator.trainable = False
        o1 = image_discriminator([img_in, gen_out])

        loss = wasserstein_loss if self.use_wasserstein else "binary_crossentropy"

        self.model = Model([img_in], [gen_out, o1])
        self.model.compile(
                loss=["mae", loss],
                loss_weights=[1., 1.],
                optimizer=self.getOptimizer())

        self.generator = Model([img_in], [gen_out])
        self.generator.compile(
                loss=["logcosh"],
                optimizer=self.getOptimizer())

        image_discriminator.summary()

        return self.model, self.model, None, [img_in], enc


    def _makeImageDiscriminator(self, img_shape):
        '''
        create image-only encoder to extract keypoints from the scene.

        Params:
        -------
        img_shape: shape of the image to encode
        '''
        img = Input(img_shape,name="img_encoder_in")
        img0 = Input(img_shape,name="img0_encoder_in")
        ins = [img, img0]
        dr = 0.3

        if self.use_wasserstein:
            loss = wasserstein_loss
            activation = "linear"
        else:
            loss = "binary_crossentropy"
            activation = "sigmoid"

        x = AddConv2D(img, 32, [4,4], 1, 0, "same", lrelu=True, bn=False)
        x0 = AddConv2D(img0, 32, [4,4], 1, 0, "same", lrelu=True, bn=False)
        x = Add()([x, x0])
        #x = Concatenate(axis=-1)([img0, img])
        x = AddConv2D(x, 32, [4,4], 2, dr, "same", lrelu=True, bn=False)
        x = AddConv2D(x, 64, [4,4], 2, dr, "same", lrelu=True, bn=False)
        x = AddConv2D(x, 128, [4,4], 2, dr, "same", lrelu=True, bn=False)
        #x = AddConv2D(x, 256, [4,4], 2, dr, "same", lrelu=True, bn=False)
        x = AddConv2D(x, 1, [1,1], 1, dr, "same", activation=activation, bn=False)
        x = AveragePooling2D(pool_size=(12,16))(x)
        x = Flatten()(x)
        #x = Dense(1, activation="linear")(x)
        discrim = Model(ins, x, name="image_discriminator")
        discrim.compile(loss=loss,
                optimizer=self.getOptimizer())
        self.image_discriminator = discrim
        return discrim

    def _getData(self, image, *args, **kwargs):
        I = np.array(image) / 255.
        return [I], [I]
