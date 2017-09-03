from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers import Lambda
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam

'''
PLANNER MODEL TOOLS
-------------------

This file contains models for performing hierarchical planner operations.


Returns for all tools:
--------
out: an output tensor
'''

def CombineArmAndGripper(arm_in, gripper_in):
    robot = Concatenate(axis=-1)([arm_in, gripper_in])
    return robot

def TileArmAndGripper(x, arm_in, gripper_in, tile_width, tile_height,
        option=None, option_in=None,
        time_distributed=None):
    arm_size = int(arm_in.shape[-1])
    gripper_size = int(gripper_in.shape[-1])

    # handle error: options and grippers
    if option is None and option_in is not None \
        or option is not None and option_in is None:
            raise RuntimeError('must provide both #opts and input')

    # generate options and tile things together
    if option is None:
        robot = CombineArmAndGripper(arm_in, gripper_in)
        reshape_size = arm_size+gripper_size
    else:
        robot = Concatenate(axis=-1)([arm_in, gripper_in, option_in])
        reshape_size = arm_size+gripper_size+option

    # time distributed or not
    if time_distributed is not None and time_distributed > 0:
        tile_shape = (1, 1, tile_width, tile_height, 1)
        robot = Reshape([time_distributed, 1, 1, reshape_size])(robot)
    else:
        tile_shape = (1, tile_width, tile_height, 1)
        robot = Reshape([1, 1, reshape_size])(robot)

    # finally perform the actual tiling
    robot = Lambda(lambda x: K.tile(x, tile_shape))(robot)
    x = Concatenate(axis=-1)([x,robot])

    return x

def GetImageEncoder(img_shape, dim, dropout_rate,
        filters, dropout=True, leaky=True,
        dense=True, flatten=True,
        layers=2,
        kernel_size=[3,3],
        time_distributed=0,):

    if time_distributed <= 0:
        ApplyTD = lambda x: x
        height4 = img_shape[0]/4
        width4 = img_shape[1]/4
        height2 = img_shape[0]/2
        width2 = img_shape[1]/2
        height = img_shape[0]
        width = img_shape[1]
        channels = img_shape[2]
    else:
        ApplyTD = lambda x: TimeDistributed(x)
        height4 = img_shape[1]/4
        width4 = img_shape[2]/4
        height2 = img_shape[1]/2
        width2 = img_shape[2]/2
        height = img_shape[1]
        width = img_shape[2]
        channels = img_shape[3]

    samples = Input(shape=img_shape)

    '''
    Convolutions for an image, terminating in a dense layer of size dim.
    '''

    if leaky:
        relu = lambda: LeakyReLU(alpha=0.2)
    else:
        relu = lambda: Activation('relu')

    x = samples

    x = ApplyTD(Conv2D(filters,
                kernel_size=kernel_size, 
                strides=(1, 1),
                padding='same'))(x)
    x = ApplyTD(relu())(x)
    if dropout:
        x = ApplyTD(Dropout(dropout_rate))(x)

    for i in range(layers):

        x = ApplyTD(Conv2D(filters,
                   kernel_size=kernel_size, 
                   strides=(2, 2),
                   padding='same'))(x)
        x = ApplyTD(relu())(x)
        if dropout:
            x = ApplyTD(Dropout(dropout_rate))(x)

    if flatten or dense:
        x = ApplyTD(Flatten())(x)
    if dense:
        x = ApplyTD(Dense(dim))(x)
        x = ApplyTD(relu())(x)

    return [samples], x

def SliceImageHypotheses(image_shape, num_hypotheses, x):
    '''
    Slice images. When we sample a set of images, we want to maintain the
    spatial organization inherent in the inputs. This is used to split one
    output into many different hypotheses.

    Here, we assume x is an input tensor of shape:
        (w,h,c) = image_shape
        x.shape == (w,h,c*num_hypotheses)

    For reference when debugging:
        # SLICING EXAMPLE:
        import keras.backend as K
        t = K.ones((12, 3))
        t1 = t[:, :1] + 1
        t2 = t[:, 1:] - 1
        t3 = K.concatenate([t1, t2])
        print(K.eval(t3))

    Parameters:
    -----------
    image_shape: (width,height,channels)
    num_hypotheses: number of images being created
    x: tensor of shape (width,height,num_hypotheses*channels)
    '''

    size = 1.
    for dim in image_shape:
        size *= dim
    y = []
    for i in range(num_hypotheses):
        xi = x[:,:,:,(3*i):(3*(i+1))]
        xi = K.expand_dims(xi,1)
        y.append(xi)
    return K.concatenate(y,axis=1)


def GetImageDecoder(dim, img_shape,
        dropout_rate, filters, kernel_size=[3,3], dropout=True, leaky=True,
        batchnorm=True,dense=True, num_hypotheses=None,
        stride2_layers=2, stride1_layers=1):

    '''
    Initial decoder: just based on getting images out of the world state
    created via the encoder.
    '''

    height16 = img_shape[0]/16
    width16 = img_shape[1]/16
    height8 = int(img_shape[0]/8)
    width8 = int(img_shape[1]/8)
    height4 = int(img_shape[0]/4)
    width4 = int(img_shape[1]/4)
    height2 = img_shape[0]/2
    width2 = img_shape[1]/2
    nchannels = img_shape[2]

    if leaky:
        relu = lambda: LeakyReLU(alpha=0.2)
    else:
        relu = lambda: Activation('relu')

    z = Input((width8*height8*filters,),name="input_image")
    x = Reshape((width8,height8,filters))(z)
    x = Dropout(dropout_rate)(x)

    height = height4
    width = width4
    for i in range(stride2_layers):

        x = Conv2DTranspose(filters,
                   kernel_size=kernel_size, 
                   strides=(2, 2),
                   padding='same')(x)
        if batchnorm:
            x = BatchNormalization(momentum=0.9)(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

        height *= 2
        width *= 2

    for i in range(stride1_layers):
        x = Conv2D(filters, # + num_labels
                   kernel_size=kernel_size, 
                   strides=(1, 1),
                   padding="same")(x)
        if batchnorm:
            x = BatchNormalization(momentum=0.9)(x)
        x = relu()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

    if num_hypotheses is not None:
        x = Conv2D(num_hypotheses*nchannels, (1, 1), padding='same')(x)
        x = Lambda(lambda x: SliceImages(img_shape,num_hypotheses,x))(x)
    else:
        x = Conv2D(nchannels, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    ins = [z]

    return ins, x

def OneHot(size=64):
    return Lambda(lambda x: tf.one_hot(tf.cast(x, tf.int32),size))#,name="label_to_one_hot")
