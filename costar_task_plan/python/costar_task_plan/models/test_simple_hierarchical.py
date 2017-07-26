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
from dense import *
from split import *

class TestSimpleHierarchical(AbstractAgentBasedModel):
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

        super(TestSimpleHierarchical, self).__init__(*args, **kwargs)

        self.model = None
        
        self.dropout_rate = 0.5
        self.dense_size = 128
        self.lstm_size = 64
        self.num_frames = 100
        self.decoder_filters = 32
        self.dense_layers = 1
        self.lstm_layers = 1

        # this is the supervisor that tells us which action to execute
        self.supervisor = None
        # these are the low-level options
        self.options = []
        # these are the conditions that tell us whether an action is finished
        # or not
        self.conditions = []

    def _makeModel(self, features, state, action, label, example, reward,
              *args, **kwargs):

        if len(features.shape) is not 3:
            raise RuntimeError('unsupported feature size')
        if features.shape[1] is not self.num_frames:
            raise RuntimeError('did not properly create net architecture')
        num_labels = int(np.max(label)+1)
        num_features = features.shape[-1]
        action_size = action.shape[-1]
        #num_frames = features.shape[0]
        ins, x = GetEncoder(self.num_frames, num_features, self.dense_size,
                self.lstm_size, self.dense_layers, self.lstm_layers)

        for i in xrange(num_labels):
            # for later
            pass

        print "num labels =", num_labels
        label_out = LSTM(num_labels,return_sequences=True)(x)
        action_out = LSTM(action_size,return_sequences=True)(x)


        self.model = Model(ins,
                           [label_out, action_out])
        self.model.summary()
        self.model.compile(loss=['binary_crossentropy', 'mse'],
                           optimizer=self.getOptimizer())

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
        print " ------- DATA BATCHED -------- "
        print features.shape
        print state.shape
        print action.shape
        print label.shape
        print example.shape
        print reward.shape
        num_actions = int(np.max(label)+1)
        self._makeModel(features, state, action, label, example, reward)
        label = self.toOneHot2D(label, num_actions)
        self.model.fit(features, [label, action],
                nb_epochs=self.epochs)

    def plot(self,*args,**kwargs):
        pass

if __name__ == '__main__':
    data = np.load('roadworld.npz')
    sampler = TestSimpleHierarchical(
            batch_size=64,
            iter=5000,
            optimizer="adam",)
    sampler.show_iter = 100
    sampler.train(**data)
    #try:
    #    sampler.train(**data)
    #except Exception, e:
    #    print e
    sampler.plot(**data)

    while(True):
        plt.pause(0.1)
