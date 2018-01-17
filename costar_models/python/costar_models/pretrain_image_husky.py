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

class PretrainImageAutoencoderHusky(HuskyRobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PretrainImageAutoencoderHusky, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = ImageCb


    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, pose) = features
        img_shape = images.shape[1:]
        pose_size = pose.shape[-1]

        img_in = Input(img_shape,name="predictor_img_in")
        img0_in = Input(img_shape,name="predictor_img0_in")
        option_in = Input((1,), name="predictor_option_in")
        encoder = self._makeImageEncoder(img_shape)
        ins = [img_in]
        
        enc = encoder(ins)
        decoder = self._makeImageDecoder(
                    self.hidden_shape,
                    self.skip_shape,)
        out = decoder(enc)

        image_discriminator = self._makeImageEncoder(img_shape, disc=True)

        #o1 = image_discriminator(ins)
        #o2 = image_discriminator([out])

        encoder.summary()
        decoder.summary()
        image_discriminator.summary()

        ae = Model(ins, [out]) #, o1, o2])
        ae.compile(
                loss=["mae"],# + ["categorical_crossentropy"]*2,
                #loss_weights=[1.,1.e-2,1e-4],
                optimizer=self.getOptimizer())
        ae.summary()
    
        return ae, ae, None, [img_in], enc

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, oin, q_target] = features
        #o1 = targets[1]
        #oin_1h = np.squeeze(self.toOneHot2D(oin, self.num_options))
        return [I], [I]#, oin_1h, oin_1h]

