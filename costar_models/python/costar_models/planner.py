
import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

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
