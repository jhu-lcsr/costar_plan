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

from .callbacks import *
from .multi_sampler import *

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

class PretrainImageGan(RobotMultiPredictionSampler):

    def __init__(self, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PretrainImageGan, self).__init__(*args, **kwargs)
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
                encoder.load_weights(self.makeName(
                    "pretrain_image_encoder",
                    "image_encoder"))
                decoder.load_weights(self.makeName(
                    "pretrain_image_encoder",
                    "image_decoder"))
            except Exception as e:
                print(">> Failed to load pretrained generator weights.")

        gen_out = decoder(enc)
        image_discriminator = self._makeImageDiscriminator(img_shape)
        self.discriminator = image_discriminator

        image_discriminator.trainable = False
        o1 = image_discriminator([img_in, gen_out])

        loss = wasserstein_loss if self.use_wasserstein else "binary_crossentropy"
        weights = [1., 1.] if self.use_wasserstein else [100., 1.]
        self.model = Model([img_in], [gen_out, o1])
        self.model.compile(
                loss=["mae", loss],
                loss_weights=weights,
                optimizer=self.getOptimizer())

        self.generator = Model([img_in], [gen_out])
        self.generator.compile(
                loss=["logcosh"],
                optimizer=self.getOptimizer())

        image_discriminator.summary()

        return self.model, self.model, None, [img_in], enc

    def _getData(self, *args, **kwargs):
        features, targets = GetAllMultiData(self.num_options, *args, **kwargs)
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

        if self.use_wasserstein:
            loss = wasserstein_loss
            activation = "linear"
        else:
            loss = "binary_crossentropy"
            activation = "sigmoid"
        
        x = AddConv2D(img, 64, [4,4], 1, dr, "same", lrelu=True, bn=False)
        x0 = AddConv2D(img0, 64, [4,4], 1, dr, "same", lrelu=True, bn=False)
        x = Add()([x, x0])
        x = AddConv2D(x, 64, [4,4], 2, dr, "same", lrelu=True, bn=False)
        x = AddConv2D(x, 128, [4,4], 2, dr, "same", lrelu=True, bn=False)
        x = AddConv2D(x, 256, [4,4], 2, dr, "same", lrelu=True, bn=False)
        x = AddConv2D(x, 1, [1,1], 1, 0., "same",
                activation=activation,
                bn=False)
        #x = AveragePooling2D(pool_size=(16,16))(x)
        x = AveragePooling2D(pool_size=(8,8))(x)
        x = Flatten()(x)

        discrim = Model(ins, x, name="image_discriminator")
        self.lr *= 2.
        discrim.compile(loss=loss, loss_weights=[1.],
                optimizer=self.getOptimizer())
        self.lr *= 0.5
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
                    if self.use_wasserstein:
                        is_fake = np.ones((self.batch_size,1))
                        is_not_fake = -1 * np.ones((self.batch_size,1))
                    else:
                        is_fake = np.random.random((self.batch_size, 1)) * 0.1 + 0.9
                        is_not_fake = np.random.random((self.batch_size, 1)) * 0.1
                    res1 = self.discriminator.train_on_batch(
                            img + target, is_not_fake)
                    inputs = img + fake if isinstance(fake, list) else img + [fake]
                    res2 = self.discriminator.train_on_batch(inputs, is_fake)
                    self.discriminator.trainable = False
                    print("\rEpoch {}, {}/{}: Descrim Real loss {}, Fake loss {}".format(
                        i+1, j, self.steps_per_epoch, res1, res2), end="")

                # Accuracy tests
                img, target = next(train_generator)
                fake = self.generator.predict(img)
                inputs = img + fake if isinstance(fake, list) else img + [fake]
                results = self.discriminator.predict(inputs)
                results2 = self.discriminator.predict(img + target)
                threshold = 0. if self.use_wasserstein else 0.5
                correct = np.count_nonzero(results >= threshold)
                correct2 = np.count_nonzero(results2 < threshold)

                print("\nAccuracy Epoch {}, real acc {}, fake acc {}".format(
                    i+1, correct/float(len(results)), correct2/float(len(results2))))

                for c in callbacks:
                    c.on_epoch_end(i)
        else: # actual gan
            d_iters = 10
            totals = [0, 0, 0, 0]

            for i in range(self.epochs):
                for j in range(self.steps_per_epoch):

                    iter_for_step = d_iters * 10 if j % 40 == 0 else d_iters
                    if not self.use_wasserstein:
                        iter_for_step = 1

                    # Discriminator loops
                    for d in range(iter_for_step):

                        # Clip the weights for the wasserstein gan
                        if self.clip_weights > 0:
                            c = self.clip_weights
                            for l in self.discriminator.layers:
                                weights = l.get_weights()
                                weights = [np.clip(w, -c, c) for w in weights]
                                l.set_weights(weights)

                        # Descriminator pass
                        img, target = next(train_generator)
                        fake = self.generator.predict(img)
                        self.discriminator.trainable = True
                        if self.use_wasserstein:
                            is_fake = np.ones((self.batch_size,1))
                            is_not_fake = -1 * np.ones((self.batch_size,1))
                        else:
                            is_fake = np.random.random((self.batch_size, 1)) * 0.1 + 0.9
                            is_not_fake = np.random.random((self.batch_size, 1)) * 0.1
                        res1 = self.discriminator.train_on_batch(
                                img + target, is_not_fake)

                        inputs = img + fake if isinstance(fake, list) else img + [fake]
                        res2 = self.discriminator.train_on_batch(inputs, is_fake)
                        print("D real loss[{}], fake loss[{}]".format(res1, res2))

                    self.discriminator.trainable = False

                    # Generator pass
                    img, target = next(train_generator)
                    res = self.model.train_on_batch(
                            img, target + [is_not_fake]
                    )
                    print('Epoch {}, {}/{}: G loss[{}], G err[{}]'.format(
                        i+1, j, self.steps_per_epoch, res[0], res[1]))

                    totals[0] += res1
                    totals[1] += res2
                    totals[2] += res[0]
                    totals[3] += res[1]


                # Accuracy tests
                img, target = next(train_generator)
                fake = self.generator.predict(img)
                inputs = img + fake if isinstance(fake, list) else img + [fake]
                results = self.discriminator.predict(inputs)
                results2 = self.discriminator.predict(img + target)
                threshold = 0. if self.use_wasserstein else 0.5
                correct = np.count_nonzero(results >= threshold)
                correct2 = np.count_nonzero(results2 < threshold)

                d_real_acc = correct / float(len(results))
                d_fake_acc = correct2 / float(len(results2))

                print("Epoch {}, testing D acc: real[{}], fake[{}]".format(
                    i, d_real_acc, d_fake_acc))

                logs = {}
                totals = [x / self.steps_per_epoch for x in totals]
                logs['D real loss'] = totals[0]
                logs['D fake loss'] = totals[1]
                logs['G loss'] = totals[2]
                logs['G l1 error'] = totals[3]
                logs['D real acc'] = d_real_acc
                logs['D fake acc'] = d_fake_acc

                for c in callbacks:
                    c.on_epoch_end(i, logs)

