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
        self.num_samples = 16
        self.trajectory_length = 48
        self.decoder_filters = 64

    def train(self, features, state, action, trace, example, reward,
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

        [features, state, action, example, trace, reward] = \
                SplitIntoChunks([features, state, action, example, trace,
                    reward],
                example, self.trajectory_length, step_size=10, padding=True,
                forward_and_back=False)

        state = state[:,:,:5]
        print "state vars =", state.shape

        # Get images for input and output from the network.
        fdata_in = FirstInChunk(features)
        sdata_in = FirstInChunk(state)

        features_size = features.shape[-1]
        state_size = state.shape[-1]

        print "-------------------------------"
        print "KEY VARIABLES:"
        print "# features =", features_size
        print "# state vars =", state_size
        print "-------------------------------"

        # Noise for sampling
        noise_in = Input((self.noise_dim,))
        features_in = Input((features_size,))
        state_in = Input((state_size,))
        #x = Concatenate()([features_in, state_in, noise_in])
        x = Concatenate()([features_in, noise_in])

        for i in xrange(3):
            x = Dense(self.dense_size)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(self.dropout_rate)(x)

        assert self.trajectory_length % 8 == 0
        x = AddSamplerLayer(x,
                int(self.num_samples),
                int(self.trajectory_length/8),
                self.decoder_filters)
        for i in xrange(2):
            x = UpSampling2D(size=(1,2))(x)
            x = Conv2D(self.decoder_filters, 3, 3, border_mode='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
        x = UpSampling2D(size=(1,2))(x)
        x = Conv2D(state_size, 3, 3, border_mode='same')(x)

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
            print "Iter %d: loss = %f"%(i,loss)

    def save(self):
        if self.model is not None:
            self.model.save_weights(self.name + ".h5f")

    def load(self):
        self.model.load_weights(self.name + ".h5f")

    def plot(self, features, state, *args, **kwargs):
        fig = plt.figure()

        for i in xrange(9):
            noise = np.random.random((1, self.noise_dim))
            trajs = self.model.predict([
                np.array([features[i*100]]),
                np.array([state[i*100]]),
                noise])
            plt.subplot(3,3,i+1)
            for j in xrange(self.num_samples):
                plt.plot(trajs[0,j,:,1],trajs[0,j,:,0])

        plt.show()

if __name__ == '__main__':
    data = np.load('test_data.npz')
    sampler = TestTrajectorySampler(
            batch_size=64,
            iter=10000,
            optimizer="adam",)
    sampler.train(**data)
    sampler.plot(**data)

