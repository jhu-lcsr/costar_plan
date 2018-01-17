from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
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
        self.load_pretrained_weights = False

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
        [img, q, g, oin, label, q_target, g_target,] = features
        return [img], [img]

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
        dr = 0
        
        x = AddConv2D(img, 64, [4,4], 1, dr, "valid", lrelu=True)
        x0 = AddConv2D(img0, 64, [4,4], 1, dr, "valid", lrelu=True)
        x = Add()([x, x0])
        x = AddConv2D(x, 64, [4,4], 2, dr, "valid", lrelu=True)
        x = AddConv2D(x, 64, [4,4], 1, dr, "same", lrelu=True)
        x = AddConv2D(x, 128, [4,4], 2, dr, "valid", lrelu=True)
        x = AddConv2D(x, 128, [4,4], 1, dr, "same", lrelu=True)
        x = AddConv2D(x, 256, [4,4], 2, dr, "valid", lrelu=True)
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

    def _fit(self, train_generator, test_generator, callbacks):

        if self.gan_method == 'mae':
            # MAE
            for i in range(self.epochs):
                for j in range(self.steps_per_epoch):
                    img, _ = next(train_generator)
                    img = img[0]
                    res = self.generator.train_on_batch(img, img)
                    print("\rEpoch {}, {}/{}: MAE loss {:.5}".format(
                        i+1, j, self.steps_per_epoch, res), end="")

                for c in callbacks:
                    c.on_epoch_end(i)

        elif self.gan_method == 'desc':
            for i in range(self.epochs):
                for j in range(self.steps_per_epoch):
                    # Descriminator pass
                    img, target = next(train_generator)
                    fake = self.generator.predict(img)
                    self.discriminator.trainable = True
                    is_fake = np.random.random((self.batch_size, 1)) * 0.1 + 0.9
                    is_not_fake = np.random.random((self.batch_size, 1)) * 0.1
                    res1 = self.discriminator.train_on_batch(
                            img + target, is_not_fake)
                    if isinstance(fake, list):
                        inputs = img + fake
                    else:
                        inputs = img + [fake]
                    res2 = self.discriminator.train_on_batch(
                            inputs, is_fake)
                    self.discriminator.trainable = False
                    print("\rEpoch {}, {}/{}: Descrim Real loss {}, Fake loss {}".format(
                        i+1, j, self.steps_per_epoch, res1, res2), end="")

                # Accuracy tests
                img, target = next(train_generator)
                fake = self.generator.predict(img)
                if isinstance(fake, list):
                    inputs = img + fake
                else:
                    inputs = img + [fake]
                results = self.discriminator.predict(inputs)
                results2 = self.discriminator.predict(img + target)
                correct = np.count_nonzero(results >= 0.5)
                correct2 = np.count_nonzero(results2 < 0.5)

                print("\nAccuracy Epoch {}, real acc {}, fake acc {}".format(
                    i+1, correct/float(len(results)), correct2/float(len(results2))))

                for c in callbacks:
                    c.on_epoch_end(i)
        else:
            for i in range(self.epochs):
                for j in range(self.steps_per_epoch):

                    # Descriminator pass
                    img, target = next(train_generator)
                    fake = self.generator.predict(img)
                    self.discriminator.trainable = True
                    is_fake = np.random.random((self.batch_size, 1)) * 0.1 + 0.9
                    is_not_fake = np.random.random((self.batch_size, 1)) * 0.1
                    res1 = self.discriminator.train_on_batch(
                            img + target, is_not_fake)
                    if isinstance(fake, list):
                        inputs = img + fake
                    else:
                        inputs = img + [fake]
                    res2 = self.discriminator.train_on_batch(
                            inputs, is_fake)
                    self.discriminator.trainable = False

                    # Generator pass
                    img, target = next(train_generator)
                    res = self.model.train_on_batch(
                            img, target + [is_not_fake]
                    )
                    print("Epoch {}, {}/{}: Gen loss {}, Gen err {}, Real loss {}, Fake loss {}".format(
                        i+1, j, self.steps_per_epoch, res[0], res[1], res1, res2))

                # Accuracy tests
                img, target = next(train_generator)
                fake = self.generator.predict(img)
                if isinstance(fake, list):
                    inputs = img + fake
                else:
                    inputs = img + [fake]
                results = self.discriminator.predict(inputs)
                results2 = self.discriminator.predict(img + target)
                correct = np.count_nonzero(results >= 0.5)
                correct2 = np.count_nonzero(results2 < 0.5)

                print("Epoch {}, real acc {}, fake acc {}".format(
                    i, correct/float(len(results)), correct2/float(len(results2))))

                for c in callbacks:
                    c.on_epoch_end(i)

