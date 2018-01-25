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
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers.merge import Add, Multiply
from keras.layers.merge import Concatenate
from keras.models import Model, Sequential

from .planner import *
from .data_utils import *

'''
Contains tools to make the sub-models for the "multi" application
'''

def _makeTrainTarget(I_target, q_target, g_target, o_target):
    if I_target is not None:
        length = I_target.shape[0]
        image_shape = I_target.shape[1:]
        image_size = 1
        for dim in image_shape:
            image_size *= dim
        image_size = int(image_size)
        Itrain = np.reshape(I_target,(length, image_size))
        return np.concatenate([Itrain, q_target,g_target,o_target],axis=-1)
    else:
        length = q_target.shape[0]
        return np.concatenate([q_target,g_target,o_target],axis=-1)

def MakeImageClassifier(model, img_shape):
    img0 = Input(img_shape,name="img0_classifier_in")
    img = Input(img_shape,name="img_classifier_in")
    bn = True
    disc = True
    dr = 0.5
    x = img
    x0 = img0

    x0 = AddConv2D(x0, 32, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 32, [5,5], 1, 0., "same", lrelu=disc, bn=bn)
    x = Add()([x0, x])

    x = AddConv2D(x, 32, [3,3], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 32, [3,3], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 32, [3,3], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [3,3], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 64, [3,3], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 128, [3,3], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 128, [3,3], 1, 0., "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 128, [3,3], 2, dr, "same", lrelu=disc, bn=bn)
    x = AddConv2D(x, 128, [3,3], 2, dr, "same", lrelu=disc, bn=bn)

    x = Flatten()(x)
    x = AddDense(x, 512, "lrelu", dr, output=True, bn=bn)
    x = AddDense(x, model.num_options, "softmax", 0., output=True, bn=False)
    image_encoder = Model([img0, img], x, name="classifier")
    image_encoder.compile(loss="categorical_crossentropy",
            optimizer=model.getOptimizer(),
            metrics=["accuracy"])
    model.classifier = image_encoder
    return image_encoder

def GetPoseModel(x, num_options, arm_size, gripper_size,
        dropout_rate=0.5, batchnorm=True):
    '''
    Make an "actor" network that takes in an encoded image and an "option"
    label and produces the next command to execute.
    '''
    xin = Input([int(d) for d in x.shape[1:]], name="pose_h_in")
    x0in = Input([int(d) for d in x.shape[1:]], name="pose_h0_in")

    x = xin
    if len(x.shape) > 2:
        # Project
        x = AddConv2D(x, 32, [3,3], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="A_project",
                constraint=None)
        x0 = AddConv2D(x, 32, [3,3], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="A0_project",
                constraint=None)
        x = Add()([x0,x])

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

    # Same setup as the state decoders
    x1 = AddDense(x, 512, "lrelu", dropout_rate, constraint=None, output=False,)
    x1 = AddDense(x1, 512, "lrelu", 0., constraint=None, output=False,)
    arm = AddDense(x1, arm_size, "linear", 0., output=True)
    gripper = AddDense(x1, gripper_size, "sigmoid", 0., output=True)
    #value = Dense(1, activation="sigmoid", name="V",)(x1)
    actor = Model([xin, option_in], [arm, gripper], name="pose")
    return actor

def GetActorModel(x, num_options, arm_size, gripper_size,
        dropout_rate=0.5, batchnorm=True):
    '''
    Make an "actor" network that takes in an encoded image and an "option"
    label and produces the next command to execute.
    '''
    xin = Input([int(d) for d in x.shape[1:]], name="actor_h_in")
    x0in = Input([int(d) for d in x.shape[1:]], name="actor_h0_in")

    option_in = Input((48,), name="actor_o_in")
    #x = Concatenate(axis=-1)([xin,x0in])
    x = xin
    if len(x.shape) > 2:
        # Project
        x = AddConv2D(x, 32, [3,3], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="A_project",
                constraint=None)
        x0 = AddConv2D(x, 32, [3,3], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="A0_project",
                constraint=None)
        x = Add()([x0,x])

        # conv down
        x = AddConv2D(x, 64, [3,3], 1, dropout_rate, "same",
                bn=batchnorm,
                lrelu=True,
                name="A_C64A",
                constraint=None)
        x = TileOnto(x, option_in, num_options, x.shape[1:3])

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

    # Same setup as the state decoders
    x1 = AddDense(x, 512, "lrelu", dropout_rate, constraint=None, output=False,)
    x1 = AddDense(x1, 512, "lrelu", 0., constraint=None, output=False,)
    arm = AddDense(x1, arm_size, "linear", 0., output=True)
    gripper = AddDense(x1, gripper_size, "sigmoid", 0., output=True)
    #value = Dense(1, activation="sigmoid", name="V",)(x1)
    actor = Model([xin, option_in], [arm, gripper], name="actor")
    return actor

def GetState(x, num_options, arm_size, gripper_size,
        dropout_rate=0.5, batchnorm=True):
    '''
    Make an "actor" network that takes in an encoded image and an "option"
    label and produces the next command to execute.
    '''
    xin = Input([int(d) for d in x.shape[1:]], name="actor_h_in")
    x0in = Input([int(d) for d in x.shape[1:]], name="actor_h0_in")
    option_in = Input((48,), name="actor_o_in")
    #x = Concatenate(axis=-1)([xin,x0in])
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
    x1 = AddDense(x1, 512, "lrelu", dropout_rate, constraint=None, output=False,)
    arm = AddDense(x1, arm_size, "linear", 0., output=True)
    gripper = AddDense(x1, gripper_size, "sigmoid", 0., output=True)
    #value = Dense(1, activation="sigmoid", name="V",)(x1)
    actor = Model([xin, option_in], [arm, gripper], name="actor")
    return actor

def MakeMultiPolicy(model, encoder, features, arm, gripper,
        arm_cmd, gripper_cmd, option):
    '''
    Create a single policy corresponding to option 

    Parameters:
    -----------
    option: index of the policy to create
    '''
    img_shape = features.shape[1:]
    arm_size = arm.shape[1]
    arm_cmd_size = arm_cmd.shape[1]
    print("arm_size ", arm_size, " arm_cmd_size ", arm_cmd_size)
    if len(gripper.shape) > 1:
        gripper_size = gripper.shape[1]
    else:
        gripper_size = 1

    img_in = Input(img_shape,name="policy_img_in")
    img0_in = Input(img_shape,name="policy_img0_in")
    arm = Input((arm_size,), name="ee_in")
    gripper = Input((gripper_size,), name="gripper_in")

    ins = [img0_in, img_in, arm, gripper]

    dr, bn = model.dropout_rate, model.use_batchnorm

    y = Concatenate()([arm, gripper])

    x = encoder(img_in)
    x0 = encoder(img0_in)

    x = Concatenate(axis=-1)([x, x0])
    x = AddConv2D(x, 32, [3,3], 1, dr, "same", lrelu=True, bn=bn)

    y = AddDense(y, 32, "relu", 0., output=True, constraint=3)
    x = TileOnto(x, y, 32, (8,8), add=False)

    x = AddConv2D(x, 32, [3,3], 1, dr, "valid", lrelu=True, bn=bn)

    x = Flatten()(x)
    #x = Concatenate()([x, arm, gripper])

    x = AddDense(x, 512, "lrelu", dr, output=True, bn=bn)
    x = AddDense(x, 512, "lrelu", dr, output=True, bn=bn)

    arm_out = Dense(arm_cmd_size, name="arm_out")(x)
    gripper_out = Dense(gripper_size, name="gripper_out")(x)

    model = Model(ins, [arm_out, gripper_out])
    model.compile(loss=model.loss, optimizer=model.getOptimizer())
    model.summary()
    return model



def GetAllMultiData(num_options, features, arm, gripper, arm_cmd, gripper_cmd, label,
            prev_label, goal_features, goal_arm, goal_gripper, value, *args, **kwargs):
        I = np.array(features) / 255. # normalize the images
        q = np.array(arm)
        g = np.array(gripper) * -1
        qa = np.array(arm_cmd)
        ga = np.array(gripper_cmd) * -1
        oin = np.array(prev_label)
        I_target = np.array(goal_features) / 255.
        q_target = np.array(goal_arm)
        g_target = np.array(goal_gripper) * -1
        o_target = np.array(label)

        # Preprocess values
        value_target = np.array(np.array(value) > 1.,dtype=float)
        #if value_target[-1] == 0:
        #    value_target = np.ones_like(value) - np.array(label == label[-1], dtype=float)
        q[:,3:] = q[:,3:] / np.pi
        q_target[:,3:] = np.array(q_target[:,3:]) / np.pi
        qa /= np.pi

        o_target_1h = np.squeeze(ToOneHot2D(o_target, num_options))
        train_target = _makeTrainTarget(
                I_target,
                q_target,
                g_target,
                o_target_1h)

        return [I, q, g, oin, label, q_target, g_target,], [
                np.expand_dims(train_target, axis=1),
                o_target,
                value_target,
                np.expand_dims(qa, axis=1),
                np.expand_dims(ga, axis=1),
                I_target]
