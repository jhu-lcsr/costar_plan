from __future__ import print_function

# ----------------------------------------
# Before importing anything else -- make sure we load the right library to save
# images to disk.
import matplotlib as mpl
mpl.use("Agg")

# ---------------------------------------
# Keras tools for creating networks
import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate, Multiply
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from .abstract import *
from .callbacks import *
from .multi_hierarchical import *
from .robot_multi_models import *
from .split import *
from .mhp_loss import *
from .loss import *
from .multi_sampler import *
from .datasets.npy_generator import *
from .parse import *

class RobotMultiKeypointsVisualizer(RobotMultiPredictionSampler):
    '''
    This loads the weights from the keypoint sampler robot and uses them to
    visualize outputs from the spatial softmax layer.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        super(RobotMultiKeypointsVisualizer, self).__init__(taskdef, *args, **kwargs)

