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

class PretrainImageAutoencoder(RobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PretrainImageAutoencoder, self).__init__(taskdef, *args, **kwargs)
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

        img0_in = Input(img_shape,name="predictor_img0_in")
        img_in = Input(img_shape,name="predictor_img_in")
        option_in = Input((1,), name="predictor_option_in")
        encoder = self._makeImageEncoder(img_shape)
        ins = [img0_in, img_in]
        if self.skip_connections:
            enc, skip = encoder(ins)
            decoder = self._makeImageDecoder(
                    self.hidden_shape,
                    self.skip_shape, True)
            out = decoder([enc, skip])
        else:
            enc = encoder(ins)
            decoder = self._makeImageDecoder(
                    self.hidden_shape,
                    self.skip_shape, False)
            out = decoder(enc)

        image_discriminator = self._makeImageEncoder(img_shape, disc=True)

        o1 = image_discriminator(ins)
        #image_discriminator.trainable = False
        o2 = image_discriminator([out, img0_in])
        o2.trainable = False

        encoder.summary()
        decoder.summary()
        image_discriminator.summary()

        ae = Model(ins, [out, o1, o2])
        ae.compile(
                loss=["mae"] + ["categorical_crossentropy"]*2,
                loss_weights=[1.,1.,0.001],
                optimizer=self.getOptimizer())
        ae.summary()
    
        return ae, ae, None, [img_in], enc

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, q_target, g_target,] = features
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        o1 = targets[1]
        oin_1h = np.squeeze(self.toOneHot2D(oin, self.num_options))
        #return [I0, I], [I, q, g, oin_1h]
        return [I0, I], [I, oin_1h, oin_1h]

