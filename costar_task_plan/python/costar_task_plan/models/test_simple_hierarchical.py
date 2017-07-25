#!/usr/bin/env python

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, Lambda
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from tensorflow import TensorShape

from abstract import AbstractAgentBasedModel
from robot_multi_models import *
from split import *

class TestSimpleHierarchal(AbstractAgentBasedModel):
    '''
    Learn a model using just some known features and an LSTM. Nothing too
    complicated.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Read in taskdef for this model (or set of models). Use it to create the
        regression neural net that we can fit to compute our next action.

        Remember, here the "labels" are computed from the task model. We can
        use images and joint states together to compute next image or next
        joint state.
        '''

        super(TestTrajectorySampler, self).__init__(*args, **kwargs)

        self.model = None
        
        self.dropout_rate = 0.5
        self.dense_size = 256
        self.num_frames = 100
        self.decoder_filters = 32
        self.dense_layers = 1

    def _makeModel(self, features, state, action, label, example, reward,
              *args, **kwargs):

        if len(features.shape) is not 2:
            raise RuntimeError('unsupported feature size')
        if features.shape[0] is not self.num_frames:
            raise RuntimeError('did not properly create net architecture')
        num_features = features.shape[-1]
        #num_frames = features.shape[0]
        ins, x = GetEncoder(self.num_frames, num_features, self.dense_size,
                self.lstm_size, self.dense_layers)


    def train(self, features, state, action, label, example, reward,
              *args, **kwargs):
        '''
        Training data -- first, break into chunks of size "trajectory_length".
        In this case we actually don't care about the action labels, which we
        will still need to extract from the task model.
        
        Instead we are just going to try to fit a distribution over
        trajectories. Right now trajectory execution is not particularly noisy,
        so this should not be super hard.

        Parameters:
        -----------
        features
        state
        action
        reward
        label
        trace

        We ignore inputs including the reward (for now!)
        '''

        print " ------- DATA -------- "
        print features.shape
        print action.shape
        print state.shape

        #state = state[:,:2]
        orig_features = features
        orig_state = state
        [features, state, action, example, label, reward] = \
                SplitIntoChunks([features, state, action, example, label,
                    reward],
                example, self.num_frames, step_size=10,
                front_padding=True,
                rear_padding=False,)
        self._makeModel(features, state, action, label, example, reward)

if __name__ == '__main__':
    data = np.load('roadworld.npz')
    sampler = TestTrajectorySampler(
            batch_size=64,
            iter=5000,
            optimizer="adam",)
    sampler.show_iter = 100
    try:
        sampler.train(**data)
    except Exception, e:
        print e
    sampler.plot(**data)

    while(True):
        plt.pause(0.1)
