from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np
import scipy

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

        img_shape = (16,16,3)
        img_in = Input(img_shape,name="predictor_img_in")
        test_in = Input(img_shape, name="descriminator_test_in")

        encoder = self._makeImageEncoder(img_shape)
        enc = encoder([img_in])
        decoder = self._makeImageDecoder(
                self.hidden_shape,
                img_shape, False)
        out = decoder(enc)


        self.lr *= 2
        image_discriminator = self._makeDiscriminator(img_shape)
        self.discriminator = image_discriminator
        self.lr *= 0.5

        image_discriminator.trainable = False
        o1 = image_discriminator([img_in, out])

        encoder.summary()
        decoder.summary()
        image_discriminator.summary()

        self.model = Model([img_in], [out, o1])
        self.model.compile(
                loss=["mae"] + ["binary_crossentropy"],
                loss_weights=[1.,0.001], #debug
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
        all = []
        for i in img:
            j = scipy.misc.imresize(i, (16,16), interp='bilinear')
            j = j.astype(float) / 255.
            all.append(j)
        img = np.array(all)

        return [img], [img]

    def _makeImageEncoder(self, img_shape):
        '''
        create image-only decoder to extract keypoints from the scene.
        
        Params:
        -------
        img_shape: shape of the image to encode
        disc: is this being created as part of a discriminator network? If so,
              we handle things slightly differently.
        '''
        x = Input(img_shape,name="img_encoder_in")
        ins = x
        dr = self.dropout_rate
        self.encoder_channels = 8
        m = 0.5
        ec = self.encoder_channels
        x = AddConv2D(x, 32, [5,5], 2, dr, "same", False, momentum=m)
        x = AddConv2D(x, 32, [5,5], 1, dr, "same", False, momentum=m)
        x = AddConv2D(x, 32, [5,5], 1, dr, "same", False, momentum=m)
        x = AddConv2D(x, 64, [5,5], 2, dr, "same", False, momentum=m)
        x = AddConv2D(x, 64, [5,5], 1, dr, "same", False, momentum=m)
        x = AddConv2D(x, 128, [5,5], 2, dr, "same", False, momentum=m)

        self.steps_down = 3
        self.hidden_dim = int(img_shape[0]/(2**self.steps_down))
        self.hidden_shape = (32,)
        x = Flatten()(x)

        # dense representation
        x = AddDense(x, self.hidden_shape[0], "lrelu", 0)
        image_encoder = Model(ins, x, name="image_encoder")
        image_encoder.compile(loss="mae", optimizer=self.getOptimizer())
        self.image_encoder = image_encoder
        return image_encoder

    def _makeDiscriminator(self, img_shape):
        '''
        create image-only decoder to extract keypoints from the scene.
        
        Params:
        -------
        img_shape: shape of the image to encode
        '''
        x = Input(img_shape,name="discriminator_in")
        y = Input(img_shape,name="discriminator_in0")
        ins = [x, y]
        dr = 0.1
        x = AddConv2D(x, 32, [5,5], 1, dr, "same", True)
        y = AddConv2D(y, 32, [5,5], 1, dr, "same", True)
        x = AddConv2D(x, 32, [5,5], 2, dr, "same", True)
        y = AddConv2D(y, 32, [5,5], 2, dr, "same", True)
        x = Concatenate(axis=-1)([x,y])

        x = AddConv2D(x, 64, [5,5], 2, dr, "same", True)
        x = AddConv2D(x, 64, [5,5], 1, dr, "same", True)
        x = AddConv2D(x, 64, [5,5], 2, dr, "same", True)
        x = AddConv2D(x, 64, [5,5], 1, dr, "same", True)

        self.steps_down = 3
        self.hidden_dim = int(img_shape[0]/(2**self.steps_down))
        x = Flatten()(x)

        x = AddDense(x, 512, "lrelu", dr, output=True)
        x = AddDense(x, 1, "sigmoid", 0, output=True)
        image_encoder = Model(ins, x, name="image_discriminator")
        image_encoder.compile(loss="mae", optimizer=self.getOptimizer())
        self.image_discriminator = image_encoder
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
        m = 0.5
        self.steps_up = 3
        hidden_dim = int(img_shape[0]/(2**self.steps_up))
        #self.tform_filters = self.encoder_channels
        (h,w,c) = (hidden_dim,
                    hidden_dim,
                    128)
        x = AddDense(x, int(h*w*c), "lrelu", 0)
        x = Reshape((h,w,c))(x)

        x = AddConv2DTranspose(x, 64, [5,5], 2, dr, momentum=m)
        x = AddConv2DTranspose(x, 64, [5,5], 1, dr, momentum=m)
        x = AddConv2DTranspose(x, 32, [5,5], 2, dr, momentum=m)
        x = AddConv2DTranspose(x, 32, [5,5], 1, dr, momentum=m)
        x = AddConv2DTranspose(x, 16, [5,5], 2, dr, momentum=m)
        x = AddConv2DTranspose(x, 16, [5,5], 1, dr, momentum=m)
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
                print("{}/{}: Gen loss {}, Real loss {}, Fake loss {}".format(
                    j, self.steps_per_epoch, res[0], res1, res2))

            # Accuracy tests
            img, _ = train_generator.next()
            img = img[0]
            fake = self.generator.predict(img)
            results = self.discriminator.predict([img, fake])
            results2 = self.discriminator.predict([img, img])
            correct = np.count_nonzero(results >= 0.5)
            correct2 = np.count_nonzero(results2 < 0.5)
            for c in callbacks:
                c.on_epoch_end(i)

            print("Epoch {}, real acc {}, fake acc {}".format(
                i, correct/float(len(results)), correct2/float(len(results2))))

