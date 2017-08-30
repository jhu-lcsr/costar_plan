#!/usr/bin/env python

from __future__ import print_function

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
from trajectory import AddSamplerLayer, TrajectorySamplerLoss

class TestTrajectorySampler(AbstractAgentBasedModel):
    '''
    This creates an architecture that will generate a large number of possible
    trajectories that we could execute. It attempts to minimize the
    sample-based KL divergence from the target data when doing so.
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
        self.num_samples = 1
        self.trajectory_length = 16
        self.decoder_filters = 32

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

        print(" ------- DATA -------- ")
        print(features.shape)
        print(action.shape)
        print(state.shape)

        state = state[:,:2]
        orig_features = features
        orig_state = state
        [features, state, action, example, label, reward] = \
                SplitIntoChunks([features, state, action, example, label,
                    reward],
                example, self.trajectory_length,
                front_padding=False,
                rear_padding=True,)

        print("state vars =", state.shape)

        # Get images for input and output from the network.
        fdata_in = FirstInChunk(features)
        sdata_in = FirstInChunk(state)

        features_size = features.shape[-1]
        state_size = state.shape[-1]

        print("-------------------------------")
        print("KEY VARIABLES:")
        print("# features =", features_size)
        print("# state vars =", state_size)
        print("-------------------------------")

        # Noise for sampling
        noise_in = Input((self.noise_dim,))
        features_in = Input((features_size,))
        state_in = Input((state_size,))
        #x = Concatenate()([features_in, state_in])#, noise_in])
        x = Concatenate()([features_in, noise_in])
        #x = features_in

        for i in xrange(3):
            x = Dense(self.dense_size)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(self.dropout_rate)(x)

        assert self.trajectory_length % 4 == 0
        x = AddSamplerLayer(x,
                int(self.num_samples),
                int(self.trajectory_length/4),
                self.decoder_filters)
        for i in xrange(1):
            x = UpSampling2D(size=(1,2))(x)
            x = Conv2D(self.decoder_filters, kernel_size=(1,1), strides=(1,1), border_mode='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
        x = UpSampling2D(size=(1,2))(x)
        x = Conv2D(state_size, kernel_size=(1,1), strides=(1,1), border_mode='same')(x)

        # s0 is the initial state. it needs to be repeated num_samples *
        # traj_length times.
        s0 = Reshape((1,1,state_size))(state_in)
        s0 = K.tile(s0,
                TensorShape([1,self.num_samples,self.trajectory_length,1]))

        # Integrate along the trajectories
        x = Lambda(lambda x: K.cumsum(x, axis=2) + s0)(x)

        state_loss = TrajectorySamplerLoss(self.num_samples,
                    self.trajectory_length, state_size)

        self.model = Model([features_in, state_in, noise_in], x)
        self.model.summary()
        self.model.compile(optimizer=self.getOptimizer(), 
                loss=state_loss)

        if self.show_iter > 0:
            fig = plt.figure()

        for i in xrange(self.iter):
            idx = np.random.randint(0, fdata_in.shape[0], size=self.batch_size)
            xf = fdata_in[idx]
            xs = sdata_in[idx]

            # create targets
            y_shape = (self.batch_size,1)+state.shape[1:]
            ya = np.reshape(state[idx],y_shape)
            
            # duplicate
            ya = ya[:,np.zeros((self.num_samples,),dtype=int)]

            noise = np.random.random((self.batch_size, self.noise_dim))
            loss = self.model.train_on_batch([xf, xs, noise], ya)
            print("Iter %d: loss = %f"%(i,loss))


            if self.show_iter > 0 and (i+1) % self.show_iter == 0:
                plt.clf()
                self.plot(orig_features, orig_state)

    def save(self):
        if self.model is not None:
            self.model.save_weights(self.name + ".h5f")

    def load(self):
        self.model.load_weights(self.name + ".h5f")

    def plot(self, features, state, *args, **kwargs):

        for i in xrange(9):
            noise = np.random.random((1, self.noise_dim))
            trajs = self.model.predict([
                np.array([features[i*100]]),
                np.array([state[i*100]]),
                noise])
            plt.subplot(3,3,i+1)
            for j in xrange(self.num_samples):
                plt.plot(trajs[0,j,:,1],trajs[0,j,:,0])
            for j in xrange(trajs.shape[2]):
                plt.plot(state[(i*100)+j][1],state[(i*100)+j][0],'*')

        plt.ion()
        plt.show(block=False)
        plt.pause(0.01)

if __name__ == '__main__':
    data = np.load('roadworld.npz')
    sampler = TestTrajectorySampler(
            batch_size=64,
            iter=5000,
            optimizer="adam",)
    sampler.show_iter = 100
    try:
        sampler.train(**data)
    except Exception as e:
        print(e)
    sampler.plot(**data)

    while(True):
        plt.pause(0.1)
