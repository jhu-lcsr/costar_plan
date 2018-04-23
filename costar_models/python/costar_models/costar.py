from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np
import tensorflow as tf

from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input
from keras.layers import BatchNormalization, Dropout
from keras.layers.noise import AlphaDropout
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers.merge import Add, Multiply
from keras.layers.merge import Concatenate
from keras.models import Model, Sequential

from .planner import *
from .data_utils import *

'''
Contains tools to make the sub-models for data collected using the CoSTAR stack
for real robot execution.
'''

def MakeCostarImageClassifier(model, img_shape, trainable=True):
    img0 = Input(img_shape,name="img0_classifier_in")
    img = Input(img_shape,name="img_classifier_in")
    bn = model.use_batchnorm and False
    disc = True
    dr = model.dropout_rate
    x = img
    x0 = img0

    #x = AddConv2D(x, 32, [7,7], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 32, [5,5], 2, 0., "same", lrelu=disc, bn=True)
    x = Dropout(dr)(x)
    #x = AddConv2D(x, 32, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    #x = AddConv2D(x, 32, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [5,5], 2, 0., "same", lrelu=disc, bn=bn)
    x = Dropout(dr)(x)
    #x = AddConv2D(x, 64, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 128, [5,5], 2, 0., "same", lrelu=disc, bn=bn)
    x = Dropout(dr)(x)
    #x = AddConv2D(x, 128, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 128, [5,5], 2, 0., "same", lrelu=disc, bn=bn)

    x = Flatten()(x)
    #x = Dropout(0.5)(x)
    #x = AddDense(x, 1024, "lrelu", 0., output=True, bn=False)
    x = Dropout(0.5)(x)
    x = AddDense(x, model.num_options, "softmax", 0., output=True, bn=False)
    image_encoder = Model([img0, img], x, name="classifier")
    if not trainable:
        image_encoder.trainable = False
    image_encoder.compile(loss="categorical_crossentropy",
            optimizer=model.getOptimizer(),
            metrics=["accuracy"])
    model.classifier = image_encoder
    return image_encoder


