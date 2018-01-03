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
                loss=["logcosh"] + ["binary_crossentropy"],
                loss_weights=[0., 1],
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
        [img, q, g, oin, q_target, g_target,] = features
        return [img], [img]

    def _makeImageEncoder(self, img_shape):
        '''
        create image-only encoder to extract keypoints from the scene.
        
        Params:
        -------
        img_shape: shape of the image to encode
        '''
        img = Input(img_shape,name="img_encoder_in")
        dr = self.dropout_rate
        x = img
        self.encoder_channels = 8
        ec = self.encoder_channels
        ins = img
        x = AddConv2D(x, 32, [5,5], 2, dr, "same")

        x = AddConv2D(x, 32, [5,5], 1, dr, "same")
        x = AddConv2D(x, 32, [5,5], 1, dr, "same")
        x = AddConv2D(x, 64, [5,5], 2, dr, "same")
        x = AddConv2D(x, 64, [5,5], 1, dr, "same")
        x = AddConv2D(x, 128, [5,5], 2, dr, "same")
        # compressing to encoded format
        x = AddConv2D(x, ec, [1,1], 1, 0, "same")

        self.steps_down = 3
        self.hidden_dim = int(img_shape[0]/(2**self.steps_down))
        #self.tform_filters = self.encoder_channels
        self.hidden_shape = (256,)
        x = Flatten()(x)
        #self.hidden_shape = (self.hidden_dim,self.hidden_dim,self.encoder_channels)

        # dense representation
        x = AddDense(x, self.hidden_shape[0], "relu", dr)
        image_encoder = Model(ins, x, name="image_encoder")
        image_encoder.compile(loss="logcosh", optimizer=self.getOptimizer())
        self.image_encoder = image_encoder
        return image_encoder

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
        dr = self.dropout_rate
        dr = 0 # 0.5
        x = Concatenate(axis=-1)([img, img0])
        x = AddConv2D(x, 64, [5,5], 2, dr, "same", lrelu=True)
#        x = AddConv2D(x, 64, [5,5], 1, dr, "same", lrelu=True)
#        x = AddConv2D(x, 64, [5,5], 1, dr, "same", lrelu=True)
        x = AddConv2D(x, 128, [5,5], 2, dr, "same", lrelu=True)
#        x = AddConv2D(x, 128, [5,5], 1, dr, "same", lrelu=True)
        x = AddConv2D(x, 256, [5,5], 2, dr, "same", lrelu=True)

        x = Flatten()(x)

        #x = AddDense(x, 512, "lrelu", dr, output=True)
        x = AddDense(x, 1, "sigmoid", 0, output=True)
        discrim = Model(ins, x, name="image_discriminator")
        discrim.compile(loss="logcosh", loss_weights=[1],
                optimizer=self.getOptimizer())
        self.image_discriminator = discrim
        return discrim

    def _makeImageDecoder(self, hidden_shape, img_shape=None, skip=False):
        '''
        helper function to construct a decoder that will make images.

        parameters:
        -----------
        img_shape: shape of the image, e.g. (64, 64, 3)
        '''
        rep = Input(hidden_shape, name="decoder_hidden_in")

        x = rep
        if self.hypothesis_dropout:
            dr = self.decoder_dropout_rate
        else:
            dr = 0.
        m = 0.99
        self.steps_up = 3
        hidden_dim = int(img_shape[0]/(2**self.steps_up))
        extra_dim = 2 * hidden_shape[0] / (hidden_dim * hidden_dim)
        (h,w,c) = (hidden_dim,
                    hidden_dim,
                    extra_dim)
        x = AddDense(x, int(h*w*c), "relu", dr)
        x = Reshape((h,w,c))(x)

#        x = AddConv2DTranspose(x, 128, [1,1], 1, 0.*dr, momentum=m)
        x = AddConv2DTranspose(x, 128, [5,5], 1, dr, momentum=m)
        x = AddConv2DTranspose(x, 64, [5,5], 2, dr, momentum=m)
        x = AddConv2DTranspose(x, 64, [5,5], 1, dr, momentum=m)
        x = AddConv2DTranspose(x, 32, [5,5], 2, dr, momentum=m)
        x = AddConv2DTranspose(x, 32, [5,5], 1, dr, momentum=m)
        x = AddConv2DTranspose(x, 32, [5,5], 2, dr, momentum=m)
        x = AddConv2DTranspose(x, 32, [5,5], 1, dr, momentum=m)
        ins = rep
        x = Conv2D(3, kernel_size=[1,1], strides=(1,1),name="convert_to_rgb")(x)
        x = Activation("sigmoid")(x)
        decoder = Model(ins, x, name="image_decoder")
        decoder.compile(loss="logcosh",optimizer=self.getOptimizer())
        self.image_decoder = decoder
        return decoder

    def _fit(self, train_generator, test_generator, callbacks):

        """
        for i in range(self.pretrain_iter):
            # Descriminator pass
            img, _ = train_generator.next()
            img = img[0]
            fake = self.generator.predict(img)
            self.discriminator.trainable = True
            is_fake = np.random.random((self.batch_size, 1)) * 0.1 + 0.9
            is_not_fake = np.random.random((self.batch_size, 1)) * 0.1
            res1 = self.discriminator.train_on_batch(
                    [img, img], is_not_fake)
            res2 = self.discriminator.train_on_batch(
                    [img, fake], is_fake)
            self.discriminator.trainable = False
            print("\rPretraining {}/{}: Real loss {}, Fake loss {}".format(
                i, self.pretrain_iter, res1, res2))
        """

        if self.gan_method == 'mae':
            # MAE
            for i in range(self.epochs):
                for j in range(self.steps_per_epoch):
                    img, _ = train_generator.next()
                    img = img[0]
                    res = self.generator.train_on_batch(img, img)
                    print("\rEpoch {}, {}/{}: MAE loss {:.5}".format(
                        i, j, self.steps_per_epoch, res), end="")
                for c in callbacks:
                    c.on_epoch_end(i)
        elif self.gan_method == 'desc':
            for i in range(self.epochs):
                for j in range(self.steps_per_epoch):
                    # Descriminator pass
                    img, _ = train_generator.next()
                    img = img[0]
                    fake = self.generator.predict(img)
                    self.discriminator.trainable = True
                    is_fake = np.random.random((self.batch_size, 1)) * 0.1 + 0.9
                    is_not_fake = np.random.random((self.batch_size, 1)) * 0.1
                    res1 = self.discriminator.train_on_batch(
                            [img, img], is_not_fake)
                    res2 = self.discriminator.train_on_batch(
                            [img, fake], is_fake)
                    self.discriminator.trainable = False
                    print("Epoch {}, {}/{}: Descrim Real loss {}, Fake loss {}".format(
                        i, j, self.steps_per_epoch, res1, res2))
                if self.save:
                    for c in callbacks:
                        c.on_epoch_end(i)
        else:
            for i in range(self.epochs):
                for j in range(self.steps_per_epoch):

                    # Descriminator pass
                    img, _ = train_generator.next()
                    img = img[0]
                    fake = self.generator.predict(img)
                    self.discriminator.trainable = True
                    is_fake = np.random.random((self.batch_size, 1)) * 0.1 + 0.9
                    is_not_fake = np.random.random((self.batch_size, 1)) * 0.1
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
                    print("Epoch {}, {}/{}: Gen loss {}, Real loss {}, Fake loss {}".format(
                        i, j, self.steps_per_epoch, res[0], res1, res2))

                # Accuracy tests
                img, _ = train_generator.next()
                img = img[0]
                fake = self.generator.predict(img)
                results = self.discriminator.predict([img, fake])
                results2 = self.discriminator.predict([img, img])
                correct = np.count_nonzero(results >= 0.5)
                correct2 = np.count_nonzero(results2 < 0.5)
                if self.save:
                    for c in callbacks:
                        c.on_epoch_end(i)

                print("Epoch {}, real acc {}, fake acc {}".format(
                    i, correct/float(len(results)), correct2/float(len(results2))))

