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
from .split import *
from .mhp_loss import *
from .loss import *
from .multi_sampler import *
from .pretrain_image import *

from .dvrk import *

class PretrainImageJigsaws(PretrainImageAutoencoder):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PretrainImageJigsaws, self).__init__(taskdef, *args, **kwargs)
        self.num_generator_files = 1

    def _makePredictor(self, image):
        '''
        Create model to predict possible manipulation goals.
        '''
        img_shape = image.shape[1:]

        img_in = Input(img_shape,name="predictor_img_in")
        encoder = MakeJigsawsImageEncoder(self, img_shape)
        ins = [img_in]
        
        enc = encoder(ins)
        decoder = MakeJigsawsImageDecoder(
                    self,
                    self.hidden_shape,
                    self.skip_shape,)
        out = decoder(enc)

        image_discriminator = MakeJigsawsImageClassifier(self, img_shape)
        image_discriminator.load_weights(
                self._makeName("discriminator_model", "predictor_weights.h5f"))
        image_discriminator.trainable = False
        o2 = image_discriminator([out])

        encoder.summary()

        ae = Model(ins, [out, o2])
        ae.compile(
                loss=["mae", "categorical_crossentropy"],
                loss_weights=[1.,1e-4],
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
        self.predictor, self.train_predictor, self.actor, ins, hidden = \
            self._makePredictor(
                image)
        if self.train_predictor is None:
            raise RuntimeError('did not make trainable model')

    def _getData(self, image, *args, **kwargs):
        I = np.array(image) / 255.
        o1 = np.array(label)
        o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
        return [I], [I, o1_1h]