from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np
import tensorflow as tf

from keras.constraints import maxnorm
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers import Lambda
from keras.layers.merge import Add, Multiply
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.constraints import max_norm

from .planner import *
from .data_utils import *

'''
===============================================================================
Contains tools to make the sub-models for the Husky application
===============================================================================
'''

def HuskyNumOptions():
    return 5

def HuskyNullOption():
    return 4

def GetHuskyActorModel(x, num_options, pose_size,
        dropout_rate=0.5, batchnorm=True):
    '''
    Make an "actor" network that takes in an encoded image and an "option"
    label and produces the next command to execute.
    '''
    xin = Input([int(d) for d in x.shape[1:]], name="actor_h_in")
    #x = Concatenate(axis=-1)([xin,x0in])
    #x0in = Input([int(d) for d in x.shape[1:]], name="actor_h0_in")
    option_in = Input((num_options,), name="actor_o_in")
    x = xin
    if len(x.shape) > 2:
        # Project
        x = AddConv2D(x, 32, [3,3], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="A_project",
                constraint=None)

        x = TileOnto(x, option_in, num_options, x.shape[1:3])

        # conv down
        x = AddConv2D(x, 64, [3,3], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="A_C64A",
                constraint=None)
        # conv across
        x = AddConv2D(x, 64, [3,3], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="A_C64B",
                constraint=None)


        x = AddConv2D(x, 32, [3,3], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="A_C32A",
                constraint=None)
        # This is the hidden representation of the world, but it should be flat
        # for our classifier to work.
        x = Flatten()(x)

    x = Concatenate()([x, option_in])

    # Same setup as the state decoders
    x1 = AddDense(x, 512, "lrelu", dropout_rate, constraint=None, output=False,)
    x1 = AddDense(x1, 512, "lrelu", 0., constraint=None, output=False,)
    pose = AddDense(x1, pose_size, "linear", 0., output=True)
    #value = Dense(1, activation="sigmoid", name="V",)(x1)
    actor = Model([xin, option_in], [pose], name="actor")
    return actor

def GetPolicyHuskyData(num_options, image, pose, action, label):
    I = np.array(image) / 255.
    p = np.array(pose)
    a = np.array(action)

def GetConditionalHuskyData(do_all, num_options, image, pose, action, label,
        prev_label, goal_image, goal_pose, value, *args, **kwargs):
    I = np.array(image) / 255.
    p = np.array(pose)
    a = np.array(action)
    I_target = np.array(goal_image) / 255.
    q_target = np.array(goal_pose)
    oin = np.array(prev_label)
    o1 = np.array(label)
    v = np.array(np.array(value) > 1.,dtype=float)

    I_target2, o2 = GetNextGoal(I_target, o1)
    I0 = I[0,:,:,:]
    length = I.shape[0]
    I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
    oin_1h = np.squeeze(ToOneHot2D(oin, num_options))

    if do_all:
        o1_1h = np.squeeze(ToOneHot2D(o1, num_options))
        return [I0, I, o1, o2, oin], [ I_target, I_target2,
                o1_1h,
                v,
                action]
    else:
        return [I0, I, o1, o2, oin], [I_target, I_target2]
