from abstract import AbstractAgentBasedModel


import keras.backend as K
import numpy as np

from tensorflow import TensorShape
from keras.layers import Dense, Reshape

class TrajectorySamplerNetwork(AbstractAgentBasedModel):
    '''
    Supervised model. Takes in a set of trajectories from the current state;
    learns a distribution that will regenerate these given some source of
    noise.

    Essentially, our goal is to minimize the average error between the whole
    set of trajectories and our samples.
    '''

    def __init__(self):
        pass

def AddSamplerLayer(x, num_samples, traj_length, feature_size, activation=None):
    '''
    Size of x must be reasonable. This turns the dense input into something
    reasonable.

    Parameters:
    x: input tensor
    num_samples: number of trajectories to generate
    traj_length: how many points we want to sample in each trajectory
    feature_size: dimensionality of each trajectory point
    activation: optional activation function to add
    '''
    x = Dense(num_samples * traj_length * feature_size)(x)
    if activation is not None:
        x = activation(x)
    x = Reshape((num_samples, traj_length, feature_size))(x)
    return x

class TrajectorySamplerLoss(object):

    def __init__(self, num_samples, traj_length, feature_size, acc=None):
        self.num_samples = num_samples
        self.traj_length = traj_length
        self.feature_size = feature_size
        self.acc = acc
        self.__name__ = "trajectory_sampler_loss"

    def __call__(self, target, pred):
        '''
        Pred must be of size:
            [batch_size=None, num_samples, traj_length, feature_size]
        Targets must be of size:
            [batch_size=None, traj_length, feature_size]

        You can use the tools in "split" to generate this sort of data (for
        targets). The actual loss function is just the L2 norm between each
        point.
        '''

        # NOTE: cannot tile here, because target and pred have to be the same
        # size. THAKS A LOT, KERAS.
        # Tile each example point by the total number of samples
        # target = K.tile(target, TensorShape([1,self.num_samples,1,1]))

        # Compute L2 norm...
        x = K.square(target - pred)
        # sum along each output dimension for each point
        x = K.sum(x,axis=-1,keepdims=False)
        # square root and sum along each trajectory
        x = K.sum(K.sqrt(x),axis=2,keepdims=False)
        # mean across each sample
        #x = K.min(x,axis=1,keepdims=False)
        x = K.mean(x,axis=1,keepdims=False) + K.min(x,axis=1,keepdims=False)

        if self.acc_cost is not None:
            # Take the L2 norm of the acceleration output and add it to the
            # loss.
            # NOTE: we may end up computing this elsewhere to avoid extra
            # penalties and stuff like that.
            #cost = K.sum(K.square(acc))
            return x + cost
        else:
            return x


