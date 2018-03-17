from __future__ import print_function

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
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from .abstract import HierarchicalAgentBasedModel
from .multi import *
from .preprocess import *
from .robot_multi_models import *

class RobotMultiHierarchical(HierarchicalAgentBasedModel):

    '''
    This is the "divide and conquer"-style classifier for training a multilevel
    model. We use our supervised action labels to learn a superviser that will
    classify which action we should be performing from any particular frame,
    and then separately we learn a model of what we should be doing at each
    frame.

    This class is set up as a SUPERVISED learning problem -- for more
    interactive training we will need to add data from an appropriate agent.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        '''
        Similarly to everything else -- we need a taskdef here.

        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(RobotMultiHierarchical, self).__init__(taskdef, *args, **kwargs)

        self.img_col_dim = 512
        self.img_num_filters = 64
        self.robot_col_dense_size = 128
        self.robot_col_dim = 64
        self.combined_dense_size = self.img_col_dim
        self.pose_col_dim = 64
        self.num_options = 48
        self.null_option = 37
        self.supervisor = None
        self.actor = None
        self.classifier = None

        # Feature presets
        self.arm_cmd_size = 6
        self.gripper_cmd_size = 1

    def _makeModel(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
        '''
        Set up all models necessary to create actions
        '''
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                features,
                arm,
                gripper)
        encoder = self._makeImageEncoder(img_shape)
        decoder = self._makeImageDecoder(self.hidden_shape)
        try:
            encoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                #"pretrain_image_gan_model",
                "image_encoder.h5f"))
            encoder.trainable = self.retrain
            decoder.load_weights(self._makeName(
                "pretrain_image_encoder_model",
                #"pretrain_image_gan_model",
                "image_decoder.h5f"))
            decoder.trainable = self.retrain
        except Exception as e:
            if not self.retrain:
                raise e

        # Make end-to-end conditional actor
        actor = self._makeConditionalActor(features, arm, gripper, arm_cmd,
                gripper_cmd, *args, **kwargs)
        self.model = actor

    def _makeSimpleActor(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
        '''
        This creates a "dumb" actor model based on a set of features.
        '''
        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        arm_cmd_size = arm_cmd.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1

        ins, x, skips = GetEncoder(
                img_shape,
                [arm_size, gripper_size],
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                pose_col_dim=self.pose_col_dim,
                discriminator=False,
                kernel_size=[3,3],
                tile=True,
                batchnorm=self.use_batchnorm,
                pre_tiling_layers=1,
                post_tiling_layers=3,
                stride1_post_tiling_layers=1)

        arm_out = Dense(arm_cmd_size, name="arm")(x)
        gripper_out = Dense(gripper_size, name="gripper")(x)

        if self.model is not None:
            raise RuntimeError('overwriting old model!')

        model = Model(ins, [arm_out, gripper_out])
        optimizer = self.getOptimizer()
        model.compile(loss=self.loss, optimizer=optimizer)
        return model

    def _makeConditionalActor(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
        '''
        This creates a "dumb" actor model based on a set of features.
        '''
        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        arm_cmd_size = arm_cmd.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1
        
        new = True
        if not new:
            ins, x, skips = GetEncoder(
                    img_shape,
                    [arm_size, gripper_size],
                    self.img_col_dim,
                    self.dropout_rate,
                    self.img_num_filters,
                    pose_col_dim=self.pose_col_dim,
                    discriminator=False,
                    kernel_size=[3,3],
                    tile=True,
                    batchnorm=self.use_batchnorm,
                    pre_tiling_layers=1,
                    post_tiling_layers=3,
                    stride1_post_tiling_layers=1,
                    option=self.num_options,
                    )
        else:
            img_in = Input(img_shape, name="ca_img_in")
            x = img_in
            x = AddConv2D(x, 64, [5,5], 2, self.dropout_rate, "valid", bn=self.use_batchnorm)
            x = AddConv2D(x, 128, [3,3], 2, self.dropout_rate, "valid", bn=self.use_batchnorm)
            x = AddConv2D(x, 128, [3,3], 1, 0., "valid", bn=self.use_batchnorm)
            x = AddConv2D(x, 128, [3,3], 1, 0., "valid", bn=self.use_batchnorm)
            x = AddConv2D(x, 128, [3,3], 1, 0., "valid", bn=self.use_batchnorm)

            arm_in = Input((arm_size,),name="ca_arm_in")
            gripper_in = Input((gripper_size,),name="ca_gripper_in")
            y = Concatenate()([arm_in, gripper_in])
            y = AddDense(y, 128, "relu", 0., output=True, constraint=3)
            x = TileOnto(x, y, 128, (8,8), add=True)

            cmd_in = Input((1,), name="option_cmd_in")
            cmd = OneHot(self.num_options)(cmd_in)
            cmd = AddDense(cmd, 128, "relu", 0., output=True, constraint=3)
            x = TileOnto(x, cmd, 128, (8,8), add=True)
            x = AddConv2D(x, 64, [3,3], 1, self.dropout_rate, "valid",
                    bn=self.use_batchnorm)
            #x = BatchNormalization()(x)
            x = Flatten()(x)
            x = AddDense(x, 512, "relu", self.dropout_rate,
                    constraint=3,
                    output=True)
            x = Dropout(self.dropout_rate)(x)
            x = AddDense(x, 512, "relu", self.dropout_rate,
                    constraint=3,
                    output=True)
            ins = [img_in, arm_in, gripper_in, cmd_in]

        arm_out = Dense(arm_cmd_size, name="arm")(x)
        gripper_out = Dense(gripper_size, name="gripper")(x)

        if self.model is not None:
            raise RuntimeError('overwriting old model!')

        model = Model(ins, [arm_out, gripper_out])
        optimizer = self.getOptimizer()
        model.compile(loss=self.loss, optimizer=optimizer)
        return model


    def _makeAll(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
        images = features
        img_shape = images.shape[1:]
        arm_size = arm.shape[-1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[-1]
        else:
            gripper_size = 1

        ins, x, skips = GetEncoder(
                img_shape,
                [arm_size, gripper_size],
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                pose_col_dim=self.pose_col_dim,
                kernel_size=[3,3],
                tile=True,
                pre_tiling_layers=1,
                post_tiling_layers=3,
                stride1_post_tiling_layers=1,
                discriminator=False,
                dense=False,
                option=self.num_options,
                flatten=False,
                )

        # =====================================================================
        # SUPERVISOR
        # Predict the next option -- does not depend on option
        for _ in range(2):
            # Repeat twice to scale down to a very small size -- this will help
            # a little with the final image layers
            x = Conv2D(int(self.img_num_filters),
                    kernel_size=[5, 5], 
                    strides=(2, 2),
                    padding='same')(x)
            x = Dropout(self.dropout_rate)(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        label_out = Dense(self.num_options, activation="softmax",name="next_option")(x)

        supervisor = Model(ins, label_out, name="supervisor")
        actor = self._makeConditionalActor(features, arm, gripper, arm_cmd,
                gripper_cmd, *args, **kwargs)

        supervisor.summary()
        print("make model setup")
        print(ins, actor.inputs)
        #model_ins = Input(name="img_in")

    def plotInfo(self, features, targets, axes):
        # debugging: plot every 5th image from the dataset
        subset = [f[range(0,25,5)] for f in features]
        data = self.predictor.predict(subset)
        for j in range(5):
            jj = j * 5
            ax = axes[1][j]
            ax.imshow(np.squeeze(data[0][j]))
            ax.axis('off')
            ax = axes[4][j]
            ax.imshow(np.squeeze(data[3][j]))
            ax.axis('off')
            ax = axes[0][j]
            ax.imshow(np.squeeze(features[0][jj]))
            ax.axis('off')
            ax = axes[2][j]
            ax.imshow(np.squeeze(targets[0][jj]))
            ax.axis('off')
            
            q0 = features[1][jj]
            q = data[1][j]
            q1 = targets[1][jj]
            ax = axes[3][j]
            ax.bar(np.arange(6),q0,1./3.,color='b')
            ax.bar(np.arange(6)+1./3.,q,1./3.,color='r')
            ax.bar(np.arange(6)+2./3.,q1,1./3.,color='g')

        plt.ion()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def _getData(self, *args, **kwargs):
        features, targets = GetAllMultiData(self.num_options, *args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        return [I, q, g, label], [np.squeeze(qa), np.squeeze(ga)]

    def _loadWeights(self, *args, **kwargs):
        '''
        Load model weights. This is the default load weights function; you may
        need to overload this for specific models.
        '''
        if self.model is not None:
            print("using " + self.name + ".h5f")
            self.model.load_weights(self.name + ".h5f")
            if self.supervisor is not None:
                try:
                    self.supervisor.load_weights(self.name + "_supervisor.h5f")
                except Exception as e:
                    print(e)
            if self.actor is not None:
                try:
                    self.actor.load_weights(self.name + "_actor.h5f")
                except Exception as e:
                    print(e)
        else:
            raise RuntimeError('_loadWeights() failed: model not found.')

    def save(self):
        '''
        Save to a filename determined by the "self.name" field.
        '''
        if self.model is not None:
            print("saving to " + self.name)
            self.model.save_weights(self.name + ".h5f")
            if self.supervisor is not None:
                self.supervisor.save_weights(self.name + "_supervisor.h5f")
            if self.actor is not None:
                self.actor.save_weights(self.name + "_actor.h5f")
            if self.classifier is not None:
                self.classifier.save_weights(self.name + "_classifier.h5f")
        else:
            raise RuntimeError('save() failed: model not found.')

    def trainFromGenerators(self, train_generator, test_generator, data=None, *args, **kwargs):
        if self.model is None:
            self._makeModel(**data)
        self.model.summary()
        self.model.fit_generator(
                train_generator,
                self.steps_per_epoch,
                epochs=self.epochs,
                validation_steps=self.validation_steps,
                validation_data=test_generator,)

    def _sizes(self, images, arm, gripper):
        img_shape = images.shape[1:]
        arm_size = arm.shape[-1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[-1]
        else:
            gripper_size = 1
        image_size = 1
        for dim in img_shape:
            image_size *= dim
        image_size = int(image_size)

        return img_shape, image_size, arm_size, gripper_size


