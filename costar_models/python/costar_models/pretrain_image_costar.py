from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from keras.layers import Input, RepeatVector, Reshape
from keras.layers.merge import Concatenate, Multiply
from keras.models import Model, Sequential

from .abstract import *
from .callbacks import *
from .robot_multi_models import *
from .mhp_loss import *
from .loss import *
from .multi_sampler import *
from .pretrain_image import *

from .plotting import *
from .multi import *
from .costar import *

class PretrainImageCostar(PretrainImageAutoencoder):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PretrainImageCostar, self).__init__(taskdef, *args, **kwargs)
        self.save_encoder_decoder = True
        self.load_jpeg = True
        self.num_generator_files = 1

    def _makePredictor(self, image):
        '''
        Create model to predict possible manipulation goals.
        '''
        img_shape = image.shape[1:]

        # Create model input
        img0_in = Input(img_shape,name="predictor_img0_in")
        img_in = Input(img_shape,name="predictor_img_in")
        ins = [img0_in, img_in]

        # Create encoder and decoder
        encoder = MakeImageEncoder(self, img_shape)
        decoder = MakeImageDecoder(
                    self,
                    self.hidden_shape,
                    self.skip_shape,)

        # Encode and connect the discriminator
        enc = encoder([img0_in, img_in])
        out = decoder(enc)

        if self.no_disc:
            ae = Model(ins, [out])
            ae.compile(
                    loss=["mae"],
                    loss_weights=[1.],
                    optimizer=self.getOptimizer())
        else:
            image_discriminator = LoadClassifierWeights(self,
                MakeCostarImageClassifier,
                img_shape)
            o2 = image_discriminator([img0_in, out])
            ae = Model(ins, [out, o2])
            ae.compile(
                    loss=["mae"] + ["categorical_crossentropy"],
                    loss_weights=[1.,1e-3],
                    optimizer=self.getOptimizer())
        ae.summary()

        return ae, ae, None, [img_in], enc

    def _makeModel(self, image, *args, **kwargs):
        '''
        Little helper function wraps makePredictor to consturct all the models.

        Parameters:
        -----------
        image, arm, gripper: variables of the appropriate sizes
        '''
        self.predictor, self.model, self.actor, ins, hidden = \
            self._makePredictor(
                image)
        if self.model is None:
            raise RuntimeError('did not make trainable model')

    # @image: array of image data for the experiment
    def _getData(self, image, label, random_draw=None, **kwargs):
        I = image

        length = len(label)
        if length == 0:
            return [], []

        # debug
        #print("Getdata: length =", length, "shape = ", I.shape, "dtype = ", I.dtype)

        I0 = np.array(I[0])
        if random_draw is None:
            o1 = np.array(label)
            I = np.array(I)
        else:
            # Randomly draw random_draw number of elements
            indexes = self._genRandomIndexes(length, random_draw)

            o1 = label[indexes]
            o1 = np.array(o1)
            I = I[indexes]
            I = np.array(I)


        #I = np.array(image) / 255.
        created_length = o1.shape[0]
        # Fill in with image 0 for every image we chose
        I0 = np.tile(np.expand_dims(I0,axis=0),[created_length,1,1,1])
        if self.no_disc:
            return [I0, I], [I]
        else:
            o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
            return [I0, I], [I, o1_1h]
