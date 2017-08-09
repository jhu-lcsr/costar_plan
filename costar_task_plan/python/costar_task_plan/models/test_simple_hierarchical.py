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
        self.dense_size = 256
        self.lstm_size = 512
        self.num_frames = 10
        self.partition_step_size = 5
        self.decoder_filters = 32
        self.dense_layers = 0
        self.lstm_layers = 3

        # this is the supervisor that tells us which action to execute
        self.supervisor = None
        # these are the low-level options
        self.options = []
        # these are the conditions that tell us whether an action is finished
        # or not
        self.conditions = []

    def _makeModel(self, features, action, label, example, reward,
              *args, **kwargs):

        if len(features.shape) is not 3:
            raise RuntimeError('unsupported feature size')
        if features.shape[1] is not self.num_frames:
            raise RuntimeError('did not properly create net architecture')
        num_labels = int(np.max(label)+1)
        num_features = features.shape[-1]
        action_size = action.shape[-1]
        #num_frames = features.shape[0]
        xin = Input((self.num_frames, num_features))
        uin = Input((self.num_frames, action_size))
        ins = [xin, uin]
        x = GetEncoder(xin, uin, self.dense_size,
                self.lstm_size, self.dense_layers, self.lstm_layers)

        ret_seq = True
        label_out = LSTM(num_labels,return_sequences=ret_seq,activation="sigmoid")(x)
        features_out = LSTM(num_features,return_sequences=ret_seq)(x)
        ok_out = LSTM(1,return_sequences=ret_seq,activation='sigmoid')(x)
        #reward_out = LSTM(1,return_sequences=True)(x)
        #action_out = LSTM(action_size,return_sequences=True)(x)

        self.model = Model(ins,
                           [label_out, features_out, ok_out])
                           #[label_out, features_out, reward_out])
        self.model.summary()
        #self.model.compile(loss=['binary_crossentropy', 'mse', 'mse'],
        self.model.compile(loss=['binary_crossentropy',
                                 'mse',
                                 'binary_crossentropy'],
                           optimizer=self.getOptimizer())

    def train(self, features, action, label, example, reward, ok,
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
        print "raw features", features.shape
        [features, action, label, reward, ok], stagger = \
                SplitIntoChunks([features, action, label,
                    reward, ok],
                example, self.num_frames, step_size=self.partition_step_size,
                front_padding=True,
                rear_padding=False,
                stagger=True,)
        [next_features, next_action, next_label, next_reward, next_ok] = stagger
        ok = np.expand_dims(ok, -1)
        next_ok = np.expand_dims(next_ok, -1)
        print "-------------"
        print "stagger comp:"
        print reward[1]
        print next_reward[1]
        print " ------- DATA BATCHED -------- "
        print "features", features.shape
        print "actions", action.shape
        print "rewards", reward.shape
        if len(reward.shape) == 2:
            reward = np.expand_dims(reward, -1)
        num_actions = int(np.max(label)+1)
        self._makeModel(features, action, label, example, reward, ok)
        label = self.toOneHot2D(label, num_actions)
        
        """
        Uncomment this to make 1d
        print "------ TO: ----------"
        label = np.squeeze(label[:,-1])
        next_features = np.squeeze(next_features[:,-1])
        next_ok = np.squeeze(next_ok[:,-1])
        print label.shape
        print next_features.shape
        print next_ok.shape
        """
        self.model.fit([features, action], [label, next_features, next_ok], epochs=self.epochs)

    def plot(self,features,action,reward,label,example,*args,**kwargs):
        plt.figure()

        # process the data
        orig_features = features
        [features, action, label, reward], stagger = \
                SplitIntoChunks([features, action, label,
                    reward],
                example, self.num_frames, step_size=self.partition_step_size,
                front_padding=True,
                rear_padding=False,
                stagger=True,)
        [next_features, next_action, next_label, next_reward] = stagger

        for i in xrange(9):
            plt.subplot(3,3,i+1)
            idx = i * 100
            x = next_features[i,:,0]
            y = next_features[i,:,1]
            plt.plot(x,y)
            label, traj, ok = self.model.predict_on_batch([
                np.array([features[i]]),
                np.array([action[i]])])
            x = traj[0][:,0]
            y = traj[0][:,1]
            #x = traj[0][0]
            #y = traj[0][1]
            plt.plot(x,y)
        plt.show()



if __name__ == '__main__':
    data = np.load('roadworld.npz')
    sampler = TestSimpleHierarchical(
            batch_size=64,
            iter=5000,
            optimizer="nadam",)
    sampler.show_iter = 100
    sampler.name = "test_dynamics"
    try:
        sampler.train(**data)
    except KeyboardInterrupt, e:
        print e
    sampler.plot(**data)
    sampler.save()

    print "done, waiting on you..."
    while(True):
        plt.pause(0.1)
