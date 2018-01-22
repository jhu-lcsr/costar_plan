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
from .multi import *
from .husky import *
from .dvrk import *

class Discriminator(RobotMultiPredictionSampler):

    def __init__(self, goal, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(Discriminator, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = None
        self.goal = goal

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, arm, gripper) = features
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                images,
                arm,
                gripper)

        disc = MakeImageClassifier(self, img_shape)
        disc.summary()
   
        return None, disc, None, None, None

    def _getData(self, features, label, goal_features, *args, **kwargs):
        I = np.array(features) / 255.
        I_target = np.array(goal_features) / 255.
        o1_1h = np.squeeze(ToOneHot2D(np.array(label), self.num_options))
        if self.goal:
            return [I_target], [o1_1h]
        else:
            return [I], [o1_1h]

class HuskyDiscriminator(RobotMultiPredictionSampler):

    def __init__(self, goal, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(HuskyDiscriminator, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = None
        self.goal = goal
        self.num_options = HuskyNumOptions()

    def _makeModel(self, image, *args, **kwargs):
        '''
        Create model to predict possible manipulation goals.
        '''
        img_shape = image.shape[1:]
        disc = MakeImageClassifier(self, img_shape)
        disc.summary()

        self.model = disc

    def _getData(self, image, goal_image, label, *args, **kwargs):
        I = np.array(image) / 255.
        I_target = np.array(goal_image) / 255.
        o1 = np.array(label)
        o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
        if self.goal:
            return [I_target], [o1_1h]
        else:
            return [I], [o1_1h]

class JigsawsDiscriminator(RobotMultiPredictionSampler):

    def __init__(self, goal, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(JigsawsDiscriminator, self).__init__(taskdef, *args, **kwargs)
        self.PredictorCb = None
        self.num_options = SuturingNumOptions()
        self.goal = goal

    def _makeModel(self, image, *args, **kwargs):
        '''
        Create model to predict possible manipulation goals.
        '''
        img_shape = image.shape[1:]
        disc = MakeJigsawsImageClassifier(self, img_shape)
        disc.summary()

        self.model = disc

    def _getData(self, image, goal_image, label, *args, **kwargs):
        I = np.array(image) / 255.
        I_target = np.array(goal_image) / 255.
        o1 = np.array(label)
        o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
        if self.goal:
            return [I_target], [o1_1h]
        else:
            return [I], [o1_1h]

