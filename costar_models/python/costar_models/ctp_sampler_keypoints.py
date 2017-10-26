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

    def __init__(self, taskdef, features, *args, **kwargs):
        super(RobotMultiKeypointsVisualizer, self).__init__(taskdef, *args, **kwargs)
        self.keypoints = None

    def _makeModel(self, features, arm, gripper, *args, **kwargs):
        '''
        Little helper function wraps makePredictor to consturct all the models.

        Parameters:
        -----------
        features, arm, gripper: variables of the appropriate sizes
        '''
        self.predictor, self.train_predictor, self.actor, ins, hidden = \
            self._makePredictor(
                (features, arm, gripper))
        self.keypoints = Model(ins, hidden, name="get_keypoints")
        self.keypoints.compile(loss="mae", optimizer=self.getOptimizer())

    def predictKeypoints(self, features):
        return self.keypoints.predict(features)[0]

    def generateImages(self, *args, **kwargs):
        '''
        Take in data representing a particular file. Generate a set of images,
        one for each frame in the file, representing the locations of important
        keypoints in the image.
        '''
        features, targets = self._getData(*args, **kwargs)
        length = features[0].shape[0]
        num_pts = self.img_col_dim/2
        for i in range(length):
            keypoints = self.predictKeypoints([np.array([f[i]]) for f in features])

            x = np.zeros((num_pts,))
            y = np.zeros((num_pts,))
            for j in range(self.img_col_dim / 2):
                jx = 2*j
                jy = 2*j + 1
                x[j] = (keypoints[jx] * 32) + 32
                y[j] = (keypoints[jy] * 32) + 32
            print('-------')
            print(x)
            print(y)
            plt.imshow(features[0][i])
            plt.plot(x,y,'*')
            plt.show()

    def generateKeypointsByOption(self, data, option, num=10):
        '''
        Generate an image with a set of keypoints, weighted by the predicted
        next option.
        '''
        pass
