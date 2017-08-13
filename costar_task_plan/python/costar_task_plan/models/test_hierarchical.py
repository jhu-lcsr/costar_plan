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

from abstract import HierarchicalAgentBasedModel
from dense import *
from split import *

class TestHierarchical(HierarchicalAgentBasedModel):
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

        super(TestHierarchical, self).__init__(*args, **kwargs)

        self.model = None
        
        self.dropout_rate = 0.5
        self.dense_size = 256
        self.lstm_size = 256
        self.num_frames = 4
        self.decoder_filters = 32
        self.filters = 64
        self.num_layers = 3 # for conv
        self.dense_layers = 1 # for dense
        self.action_layers = 2
        self.lstm_layers = 1
        self.partition_step_size = 1
        self.num_actions = 13

        self.fit_policies = True
        self.fit_baseline = False

        self.time = True

    def _makeSupervisor(self, features, label, num_labels):
        '''
        This needs to create a supervisor. This one maps from input to the
        space of possible action labels.
        '''
        if self.time:
            fin = Input(features.shape[1:])
            x = GetLSTMEncoder(fin, None, self.dense_size, self.lstm_size, self.dense_layers,
                    self.lstm_layers)
        else:
            fin = Input((features.shape[-1],))
            x = GetDenseEncoder(fin, None, self.dense_size,
                    self.dense_layers,)

        label_out = Dense(num_labels, activation="sigmoid")(x)

        supervisor = Model([fin], [label_out])
        supervisor.compile(
                loss=["binary_crossentropy"],
                optimizer=self.getOptimizer())
        return x, supervisor


    def _makePolicy(self, features, action, hidden=None):
        num_actions = action.shape[-1]
        if self.time:
            fin = Input(features.shape[1:])
            x = GetLSTMEncoder(fin, None, self.dense_size, self.lstm_size,
                    self.dense_layers,
                    self.lstm_layers)
        else:
            fin = Input((features.shape[-1],))
            x = GetDenseEncoder(fin, None, self.dense_size,
                    self.dense_layers,)

        label_out = Dense(num_actions, activation="linear")(x)

        policy = Model([fin], [label_out])
        policy.compile(
                loss=["mse"],
                optimizer=self.getOptimizer())
        return policy

    def _makeModel(self, features, state, action, label, example, reward,
              *args, **kwargs):

        '''
        Model structure:
            f(x, x0) --> h
            pi(h) --> [p(o_1), ..., p(o_n)]
        
        Condition: takes in last decision point plus option, returns one-hot
        vector indicating if that option is still active.
        Policy: takes in last decision point plus option, returns controls.
        '''
        self._makeHierarchicalModel(features, action, label,
                example, reward)

    def train(self, features, state, action, label, example, reward, ok,
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
        print "features:", features.shape, 
        print "actions:", action.shape
        print "labels:", label.shape, "max:", max(label)
        num_actions = int(np.max(label)+1)

        # Report some information on the data
        print "DATA LABELS: (max label = %d)"%(num_actions-1)
        for i in xrange(self.num_actions):
            count = np.sum(label == i)
            percent = float(count) / len(label)
            print "action %d: %.02f%% (%d/%d)"%(i,percent*100,count,len(label))

        #state = state[:,:2]
        orig_label = label
        orig_features = features
        orig_state = state
        label = np.squeeze(self.toOneHot2D(label, self.num_actions))
        assert np.all(np.argmax(label,axis=-1) == orig_label)

        if self.time:
            print "Doing data preprocessing to create chunks:"
            [features, action, label, reward, ok], stagger = \
                   SplitIntoChunks(
                    datasets=[features, action, label, ok],
                    labels=example,
                    reward=reward,
                    reward_threshold=0.,
                    chunk_length=self.num_frames,
                    step_size=self.partition_step_size,
                    front_padding=True,
                    rear_padding=False,
                    stagger=True,
                    )
            #[next_features, next_action, next_label, next_reward, next_ok] = stagger
            #features = np.expand_dims(features, -1)
            print "...done."
            labels_test = np.argmax(label,axis=-1).flatten()
            print "CHECK LABELS:"
            for i in xrange(self.num_actions):
                count = np.sum(labels_test == i)
                percent = float(count) / len(labels_test)
                print "action %d: %.02f%%(%d/%d)"%(i,percent*100,count,len(labels_test))

        print " ------- DATA BATCHED -------- "
        print "features now shape", features.shape
        print "actions now shape", action.shape
        print "labels now shape", label.shape

        label_target = np.squeeze(label[:,-1,:])
        action_target = np.squeeze(action[:,-1,:])

        self._makeModel(features, state, action, label, example, reward)
        self._fitSupervisor(features, label, label_target)
        if self.fit_policies:
            self._fitPolicies(features, label, action_target)
        if self.fit_baseline:
            self._fitBaseline(features, action_target)

    def plot(self,*args,**kwargs):
        # TODO
        pass

if __name__ == '__main__':

    data = np.load('roadworld-2018-08-09.npz')
    sampler = TestHierarchical(
            batch_size=64,
            iter=5000,
            epochs=10,
            optimizer="adam",
            task="roadworld",)

    sampler.fit_policies = True
    sampler.fit_baseline = True

    sampler.show_iter = 100
    sampler.train(**data)
    sampler.plot(**data)
    sampler.save()

