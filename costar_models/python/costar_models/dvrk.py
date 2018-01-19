from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np
import tensorflow as tf

from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers import Lambda
from keras.layers.merge import Add, Multiply
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential

from .planner import *

'''
Contains tools to make the sub-models for the DVRK application
'''


def SuturingNumOptions():
    return 15

def MakeJigsawsImageClassifier(model, img_shape):
    img = Input(img_shape,name="img_classifier_in")
    bn = False
    disc = True
    dr = 0.5 #model.dropout_rate
    x = img

    #x = BatchNormalization()(x)
    x = AddConv2D(x, 32, [7,7], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 32, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 32, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 32, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 128, [5,5], 2, dr, "same", lrelu=disc, bn=bn)

    #x = MaxPooling2D((3,4))(x)
    x = Flatten()(x)
    #x = AddDense(x, 512, "lrelu", dr, output=True, bn=bn)
    x = AddDense(x, model.num_options, "softmax", 0., output=True, bn=False)
    image_encoder = Model([img], x, name="classifier")
    image_encoder.compile(loss="categorical_crossentropy",
                          metrics="accuracy",
                          optimizer=model.getOptimizer())
    model.classifier = image_encoder
    return image_encoder



def MakeJigsawsImageEncoder(model, img_shape, disc=False):
    '''
    create image-only decoder to extract keypoints from the scene.
    
    Params:
    -------
    img_shape: shape of the image to encode
    disc: is this being created as part of a discriminator network? If so,
          we handle things slightly differently.
    '''
    img = Input(img_shape,name="img_encoder_in")
    bn = not disc and model.use_batchnorm
    #img0 = Input(img_shape,name="img0_encoder_in")
    dr = model.dropout_rate
    x = img
    #x0 = img0
    x = AddConv2D(x, 32, [7,7], 1, 0., "same", lrelu=disc, bn=bn)
    #x0 = AddConv2D(x0, 32, [7,7], 1, dr, "same", lrelu=disc, bn=bn)
    #x = Concatenate(axis=-1)([x,x0])

    x = AddConv2D(x, 32, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 32, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 32, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 128, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 128, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 128, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
    #x = AddConv2D(x, 256, [5,5], 2, dr, "same", lrelu=disc, bn=bn)

    if model.use_spatial_softmax and not disc:
        def _ssm(x):
            return spatial_softmax(x)
        model.encoder_channels = 32
        x = AddConv2D(x, model.encoder_channels, [1,1], 1, 0.*dr,
                "same", lrelu=disc, bn=bn)
        x = Lambda(_ssm,name="encoder_spatial_softmax")(x)
        model.hidden_shape = (model.encoder_channels*2,)
        model.hidden_size = 2*model.encoder_channels
        model.hidden_shape = (model.hidden_size,)
    else:
        model.encoder_channels = 8
        x = AddConv2D(x, model.encoder_channels, [1,1], 1, 0.*dr,
                "same", lrelu=disc, bn=bn)
        model.steps_down = 3
        model.hidden_dim = int(img_shape[0]/(2**model.steps_down))
        model.hidden_shape = (model.hidden_dim,model.hidden_dim,model.encoder_channels)

    if not disc:
        image_encoder = Model([img], x, name="Ienc")
        image_encoder.compile(loss="mae", optimizer=model.getOptimizer())
        model.image_encoder = image_encoder
    else:
        bnv = model.use_batchnorm
        x = Flatten()(x)
        x = AddDense(x, 512, "lrelu", dr, output=True, bn=bnv)
        x = AddDense(x, model.num_options, "softmax", 0., output=True, bn=bnv)
        image_encoder = Model([img], x, name="Idisc")
        image_encoder.compile(loss="mae", optimizer=model.getOptimizer())
        model.image_discriminator = image_encoder
    return image_encoder

def MakeJigsawsImageDecoder(model, hidden_shape, img_shape=None, copy=False):
    '''
    helper function to construct a decoder that will make images.

    parameters:
    -----------
    img_shape: shape of the image, e.g. (64,64,3)
    '''
    if model.use_spatial_softmax:
        rep = Input((model.hidden_size,),name="decoder_hidden_in")
    else:
        rep = Input(hidden_shape,name="decoder_hidden_in")

    x = rep
    dr = model.decoder_dropout_rate if model.hypothesis_dropout else 0
    bn = model.use_batchnorm
    
    if model.use_spatial_softmax:
        model.steps_up = 3
        hidden_dim = int(img_shape[0]/(2**model.steps_up))
        (h,w,c) = (hidden_dim,
                   hidden_dim,
                   model.encoder_channels)
        x = AddDense(x, int(h*w*c), "relu", dr, bn=bn)
        x = Reshape((h,w,c))(x)

    #x = AddConv2DTranspose(x, 64, [5,5], 1, dr, bn=bn)
    x = AddConv2DTranspose(x, 128, [1,1], 1, 0., bn=bn)
    x = AddConv2DTranspose(x, 128, [5,5], 2, dr, bn=bn)
    x = AddConv2DTranspose(x, 128, [5,5], 1, 0., bn=bn)
    x = AddConv2DTranspose(x, 64, [5,5], 2, dr, bn=bn)
    x = AddConv2DTranspose(x, 64, [5,5], 1, 0., bn=bn)
    x = AddConv2DTranspose(x, 32, [5,5], 2, dr, bn=bn)
    x = AddConv2DTranspose(x, 32, [5,5], 1, 0., bn=bn)
    x = AddConv2DTranspose(x, 32, [5,5], 2, dr, bn=bn)
    x = AddConv2DTranspose(x, 32, [5,5], 1, 0., bn=bn)
    ins = rep
    x = Conv2D(3, kernel_size=[1,1], strides=(1,1),name="convert_to_rgb")(x)
    x = Activation("sigmoid")(x)
    if not copy:
        decoder = Model(ins, x, name="Idec")
        decoder.compile(loss="mae",optimizer=model.getOptimizer())
        model.image_decoder = decoder
    else:
        decoder = Model(ins, x,)
        decoder.compile(loss="mae",optimizer=model.getOptimizer())
    return decoder


