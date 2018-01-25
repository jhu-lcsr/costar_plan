from __future__ import print_function

'''
===============================================================================
Contains tools to make the sub-models for the DVRK application
===============================================================================
'''

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
from .temporary import *

def SuturingNumOptions():
    return 15

def MakeJigsawsImageClassifier(model, img_shape):
    img0 = Input(img_shape,name="img0_classifier_in")
    img = Input(img_shape,name="img_classifier_in")
    bn = True
    disc = True
    dr = 0.5 #model.dropout_rate
    x = img
    x0 = img0

    x0 = AddConv2D(x0, 32, [7,7], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 32, [7,7], 1, 0., "same", lrelu=disc, bn=bn)
    x = Add()([x0, x])

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
    x = AddDense(x, 512, "lrelu", dr, output=True, bn=bn)
    x = AddDense(x, model.num_options, "softmax", 0., output=True, bn=False)
    image_encoder = Model([img], x, name="classifier")
    image_encoder.compile(loss="categorical_crossentropy",
                          metrics=["accuracy"],
                          optimizer=model.getOptimizer())
    model.classifier = image_encoder
    return image_encoder

def MakeJigsawsExpand(model, x, h_dim=(12,16)):
    '''
    Take a model and project it out to whatever size
    '''
    return AddConv2D(x, 64, [5,5], 1, 0.)

def MakeJigsawsMultiDecoder(model, decoder, num_images=4, h_dim=(12,16)):
    '''
    Make multiple images
    '''
    h = Input((h_dim[0], h_dim[1], 64),name="h_in")
    x = h

    xs = []
    for i in range(num_images):
        xi = AddConv2D(x, 64, [5, 5], stride=1,
                dropout_rate=0.)
        xi = AddConv2D(x, 64, [5, 5], stride=1,
                dropout_rate=0.)
        xi = AddConv2D(xi, model.encoder_channels, [5, 5], stride=1,
                dropout_rate=0.)
        xi = decoder(xi)
        img_x = Lambda(
            lambda y: K.expand_dims(y, 1),
            name="img_hypothesis_%d"%i)(xi)
        xs.append(img_x)
    img_out = Concatenate(axis=1)(xs)

    mm = Model(h, img_out, name="multi")
    mm.compile(loss="mae", optimizer=model.getOptimizer())

    return mm

def MakeJigsawsTransform(model, h_dim=(12,16)):
    '''
    This is the version made for the newer code, it is set up to use both
    the initial and current observed world and creates a transform
    dependent on which action you wish to perform.

    Parameters:
    -----------
    model: contains multiple sub-models and training configuration
    h_dim: hidden space default (height, width)

    Returns:
    --------
    transform model

    This will also set the "transform_model" field of "model".
    '''
    h = Input((h_dim[0], h_dim[1], 64),name="h_in")
    h0 = Input((h_dim[0],h_dim[1], model.encoder_channels),name="h0_in")
    option = Input((model.num_options,),name="t_opt_in")
    x = h # This is already encoded
    x0 = AddConv2D(h0, 64, [1,1], 1, 0.)

    # Combine the hidden state observations
    x = Concatenate()([x, x0])
    x = AddConv2D(x, 64, [5,5], 1, 0.)
    skip0 = x

    # store this for skip connection
    x = AddConv2D(x, 64, [5,5], 2, 0.)
    skip = x

    # Add dense information
    y = AddDense(option, 64, "relu", 0., constraint=None, output=False)
    x = TileOnto(x, y, 64, (h_dim[0]/2, h_dim[1]/2), add=True)
    x = AddConv2D(x, 64, [5,5], 1, 0.)

    # --- start ssm block
    def _ssm(x):
        return spatial_softmax(x)
    x = Lambda(_ssm,name="encoder_spatial_softmax")(x)
    x = AddDense(x, 128, "relu", 0.,
            constraint=None, output=False,)
    x = AddDense(x, h_dim[0] * h_dim[1] * 64/16, "relu", model.dropout_rate, constraint=None, output=False)
    x = Reshape([h_dim[0]/4, h_dim[1]/4, 64])(x)
    x = AddConv2DTranspose(x, 64, [5,5], 2, 0.)

    # --- end ssm block
    x = Concatenate()([x, skip])
    x = Dropout(model.dropout_rate)(x)
    x = AddConv2DTranspose(x, 64,
            [5,5],
            stride=2,
            dropout_rate=model.dropout_rate)

    x = Concatenate()([x, skip0])
    x = AddConv2D(x, 64,
            [5,5],
            stride=1,
            dropout_rate=model.dropout_rate)

    # --------------------------------------------------------------------
    # Put resulting image into the output shape
    #x = AddConv2D(x, model.encoder_channels, [1, 1], stride=1,
    #        dropout_rate=0.)
    model.transform_model = Model([h0,h,option], x, name="tform")
    model.transform_model.compile(loss="mae", optimizer=model.getOptimizer())
    #model.transform_model.summary()
    return model.transform_model


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
    #x = AddConv2D(x, 128, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    #x = AddConv2D(x, 128, [5,5], 2, dr, "same", lrelu=disc, bn=bn)
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
    #x = AddConv2DTranspose(x, 128, [5,5], 2, dr, bn=bn)
    #x = AddConv2DTranspose(x, 128, [5,5], 1, 0., bn=bn)
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

def GetJigsawsNextModel(x, num_options, dense_size, dropout_rate=0.5, batchnorm=True):
    '''
    Next actions
    '''

    xin = Input([int(d) for d in x.shape[1:]], name="Nx_prev_h_in")
    x0in = Input([int(d) for d in x.shape[1:]], name="Nx_prev_h0_in")
    option_in = Input((1,), name="Nx_prev_o_in")
    x = xin
    x0 = x0in
    if len(x.shape) > 2:
        # Project
        x = AddConv2D(x, 32, [1,1], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="Nx_project",
                constraint=None)
        x0 = AddConv2D(x0, 32, [1,1], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="Nx_project0",
                constraint=None)
        x = Add()([x,x0])

        if num_options > 0:
            option_x = OneHot(num_options)(option_in)
            option_x = Flatten()(option_x)
            x = TileOnto(x, option_x, num_options, x.shape[1:3])

        # conv down
        x = AddConv2D(x, 64, [3,3], 1, dropout_rate, "valid",
                bn=batchnorm,
                lrelu=True,
                name="Nx_C64A",
                constraint=None)

        x = AddConv2D(x, 32, [3,3], 1, dropout_rate, "valid",
                bn=batchnorm,
                lrelu=True,
                name="Nx_C32A",
                constraint=None)
        # This is the hidden representation of the world, but it should be flat
        # for our classifier to work.
        x = Flatten()(x)

    # Next options
    x1 = AddDense(x, dense_size, "relu", dropout_rate, constraint=None,
            output=False,)
    x1 = AddDense(x1, dense_size, "relu", 0., constraint=None,
            output=False,)

    next_option_out = Dense(num_options,
            activation="sigmoid", name="lnext",)(x1)
    next_model = Model([x0in, xin, option_in], next_option_out, name="next")
    #next_model = Model([xin, option_in], next_option_out, name="next")
    return next_model


