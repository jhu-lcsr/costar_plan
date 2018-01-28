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

from .husky_sampler import *
from .husky import *

class PretrainImageAutoencoderHusky(HuskyRobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PretrainImageAutoencoderHusky, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = ImageCb
        self.num_options = HuskyNumOptions()
        self.null_option = HuskyNumOptions()

    def _makeModel(self, image, *args, **kwargs):
        '''
        Create model to predict possible manipulation goals.
        '''
        img_shape = image.shape[1:]

        img_in = Input(img_shape,name="predictor_img_in")
        img0_in = Input(img_shape,name="predictor_img0_in")
        option_in = Input((1,), name="predictor_option_in")
        encoder = self._makeImageEncoder(img_shape)
        ins = [img0_in, img_in]
        
        enc = encoder([img_in])
        decoder = self._makeImageDecoder(
                    self.hidden_shape,
                    self.skip_shape,)
        out = decoder(enc)

        # Discriminate on distinctive features like heading we hope
        image_discriminator = LoadClassifierWeights(self,
                MakeImageClassifier,
                img_shape)
        o2 = image_discriminator([out])

        ae = Model(ins, [out, o2])
        ae.compile(
                loss=["mae", "categorical_crossentropy"],
                loss_weights=[1.,1e-4],
                optimizer=self.getOptimizer())
        ae.summary()
    
        self.predictor = ae
        self.model = ae
        self.actor = None

    def _getData(self, image, label, *args, **kwargs):
        I = np.array(image) / 255.
        o1 = np.array(label)
        o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        return [I0, I], [I, o1_1h]

