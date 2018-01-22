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

        img_in = Input(img_shape,name="predictor_img_in")
        img0_in = Input(img_shape,name="predictor_img0_in")
        option_in = Input((1,), name="predictor_option_in")
        encoder = self._makeImageEncoder(img_shape)
        ins = [img0_in, img_in]
        
        # Create the encoder
        enc = encoder(img_in)
        decoder = self._makeImageDecoder(
                    self.hidden_shape,
                    self.skip_shape,)
        out = decoder(enc)

        # Create the discriminator
        image_discriminator = MakeImageClassifier(self, img_shape)
        image_discriminator.load_weights(
                self.makeName("discriminator", "classifeir.h5f"))
        image_discriminator.trainable = False
            
        o2 = image_discriminator([img0_in, out])

        ae = Model(ins, [out, o2])
        ae.compile(
                loss=["mae"] + ["categorical_crossentropy"],
                loss_weights=[1.,1e-3],
                optimizer=self.getOptimizer())
        ae.summary()
    
        return ae, ae, None, [img_in], enc

    def _getData(self, *args, **kwargs):
        features, targets = GetAllMultiData(self.num_options, *args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        o1 = targets[1]
        oin_1h = np.squeeze(ToOneHot2D(oin, self.num_options))
        return [I, I], [I, oin_1h]

