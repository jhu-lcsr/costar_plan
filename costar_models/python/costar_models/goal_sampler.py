from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Activation, Flatten
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

class RobotMultiGoalSampler(RobotMultiPredictionSampler):

    '''
    This class is set up as a SUPERVISED learning problem -- for more
    interactive training we will need to add data from an appropriate agent.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(RobotMultiGoalSampler, self).__init__(taskdef, *args, **kwargs)

        self.num_frames = 1

        self.dropout_rate = 0.2
        self.img_col_dim = 128
        self.img_num_filters = 64
        self.tform_filters = 64
        self.combined_dense_size = 128
        self.num_hypotheses = 4
        self.num_transforms = 3
        self.validation_split = 0.1
        self.num_options = 48
        self.PredictorCb = PredictorGoals

        # Encoder architecture
        self.extra_layers = 1
        self.steps_down = 3

        self.hidden_dim = 64/(2**self.steps_down)
        self.hidden_shape = (self.hidden_dim,self.hidden_dim,self.tform_filters)

        self.predictor = None
        self.train_predictor = None
        self.actor = None

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, arm, gripper) = features
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

        ins, enc, skips, robot_skip = GetEncoder(img_shape,
                arm_size,
                gripper_size,
                self.img_col_dim,
                dropout_rate=self.dropout_rate,
                filters=self.img_num_filters,
                leaky=False,
                dropout=True,
                pre_tiling_layers=self.extra_layers,
                post_tiling_layers=self.steps_down,
                kernel_size=[3,3],
                dense=self.dense_representation,
                batchnorm=True,
                tile=True,
                flatten=False,
                output_filters=self.tform_filters,
                option=self.num_options,
                )

        img_in, arm_in, gripper_in, option_in = ins

        decoder = GetArmGripperDecoder(self.img_col_dim,
                        img_shape,
                        dropout_rate=self.dropout_rate,
                        dense_size=self.combined_dense_size,
                        dense=self.dense_representation,
                        kernel_size=[5,5],
                        filters=self.img_num_filters,
                        stride2_layers=self.steps_down,
                        stride1_layers=self.extra_layers,
                        tform_filters=self.tform_filters,
                        num_options=self.num_options,
                        arm_size=arm_size,
                        gripper_size=gripper_size,
                        dropout=True,
                        upsampling=self.upsampling_method,
                        leaky=True,
                        skips=skips,
                        robot_skip=robot_skip,
                        resnet_blocks=self.residual,
                        batchnorm=True,)


        arm_outs = []
        gripper_outs = []
        train_outs = []
        label_outs = []

        skips.reverse()
        decoder.compile(loss="mae",optimizer=self.getOptimizer())
        decoder.summary()

        z = Input((self.num_hypotheses, self.noise_dim,),name="noise_in")

        # =====================================================================
        # Create many different image decoders

        for i in range(self.num_hypotheses):
            if self.dense_representation:
                transform = GetDenseTransform(
                        dim=self.img_col_dim,
                        input_size=self.img_col_dim,
                        output_size=self.img_col_dim,
                        idx=i,
                        batchnorm=True,
                        dropout=True,
                        dropout_rate=self.dropout_rate,
                        leaky=True,
                        num_blocks=self.num_transforms,
                        relu=True,
                        resnet_blocks=self.residual,
                        use_noise=self.use_noise,
                        noise_dim=self.noise_dim,)
            else:
                transform = GetTransform(
                        rep_size=(self.hidden_dim, self.hidden_dim),
                        filters=self.tform_filters,
                        kernel_size=[3,3],
                        idx=i,
                        batchnorm=True,
                        dropout=True,
                        dropout_rate=self.dropout_rate,
                        leaky=True,
                        num_blocks=self.num_transforms,
                        relu=True,
                        resnet_blocks=self.residual,
                        use_noise=self.use_noise,
                        noise_dim=self.noise_dim,)
            if i == 0:
                transform.summary()
            if self.use_noise:
                zi = Lambda(lambda x: x[:,i], name="slice_z%d"%i)(z)
                x = transform([enc, zi])
            else:
                x = transform([enc])
            
            # This maps from our latent world state back into observable images.
            arm_x, gripper_x, label_x = decoder([x])

            # Create the training outputs
            train_x = Concatenate(axis=-1,name="combine_train_%d"%i)([
                            arm_x,
                            gripper_x,
                            label_x])
            arm_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="arm_hypothesis_%d"%i)(arm_x)
            gripper_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="gripper_hypothesis_%d"%i)(gripper_x)
            label_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="label_hypothesis_%d"%i)(label_x)
            train_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="flattened_hypothesis_%d"%i)(train_x)

            arm_outs.append(arm_x)
            gripper_outs.append(gripper_x)
            label_outs.append(label_x)
            train_outs.append(train_x)

        arm_out = Concatenate(axis=1)(arm_outs)
        gripper_out = Concatenate(axis=1)(gripper_outs)
        label_out = Concatenate(axis=1)(label_outs)
        train_out = Concatenate(axis=1,name="all_train_outs")(train_outs)

        # =====================================================================
        # Hypothesis probabilities
        #sum_p_out, p_out = GetHypothesisProbability(enc,
        #        self.num_hypotheses,
        #        self.num_options,
        #        labels=label_out,
        #        filters=self.img_num_filters,
        #        kernel_size=[5,5],
        #        dropout_rate=self.dropout_rate)
        value_out, next_option_out = GetNextOptionAndValue(enc,
                                                           self.num_options,)
        # =====================================================================
        # Training the actor policy
        y = enc
        if not self.dense_representation:
            y = Conv2D(int(self.img_num_filters/4),
                    kernel_size=[5,5], 
                    strides=(2, 2),
                    padding='same')(y)
            y = Dropout(self.dropout_rate)(y)
            y = LeakyReLU(0.2)(y)
            y = BatchNormalization(momentum=0.9)(y)
            y = Flatten()(y)
        y = Dense(self.combined_dense_size)(y)
        y = Dropout(self.dropout_rate)(y)
        y = LeakyReLU(0.2)(y)
        y = BatchNormalization(momentum=0.9)(y)
        arm_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="arm_action")(
                Dense(arm_size-1)(y))
        gripper_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="gripper_action")(
                Dense(gripper_size)(y))

        # =====================================================================
        # Create models to train
        sampler = Model(ins + [z],
                [arm_out, gripper_out, label_out, next_option_out, value_out])
        actor = Model(ins, [arm_cmd_out, gripper_cmd_out])
        train_predictor = Model(ins + [z],
                [train_out, next_option_out,
                value_out]) #, arm_cmd_out, gripper_cmd_out])

        # =====================================================================
        # Create models to train
        train_predictor.compile(
                loss=[
                    MhpLossWithShape(
                        num_hypotheses=self.num_hypotheses,
                        outputs=[arm_size, gripper_size, self.num_options],
                        weights=[0.4,0.3,0.3],
                        loss=["mae","mae","categorical_crossentropy"],
                        avg_weight=0.05),
                    "binary_crossentropy","binary_crossentropy"],
                loss_weights=[1.0,0.1,0.1,],
                optimizer=self.getOptimizer())
        sampler.compile(loss="mae", optimizer=self.getOptimizer())
        train_predictor.summary()

        return sampler, train_predictor, actor

    def _makeModel(self, features, arm, gripper, *args, **kwargs):
        '''
        Little helper function wraps makePredictor to consturct all the models.

        Parameters:
        -----------
        features, arm, gripper: variables of the appropriate sizes
        '''
        self.predictor, self.train_predictor, self.actor = \
            self._makePredictor(
                (features, arm, gripper))

    def _getAllData(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            prev_label, goal_features, goal_arm, goal_gripper, value, *args, **kwargs):
        I = features
        q = arm
        g = gripper
        qa = arm_cmd
        ga = gripper_cmd
        oin = prev_label
        I_target = goal_features
        q_target = goal_arm
        g_target = goal_gripper
        o_target = label
        value_target = np.array(value > 1.,dtype=float)

        # ==============================
        image_shape = I.shape[1:]
        image_size = 1
        for dim in image_shape:
            image_size *= dim
        image_size = int(image_size)
        arm_size = q.shape[-1]
        gripper_size = g.shape[-1]

        train_size = image_size + arm_size + gripper_size + self.num_options
        assert gripper_size == 1
        #assert train_size == 12295 + self.num_options
        # NOTE: arm size is one bigger when we have quaternions
        #assert train_size == 12296 + self.num_options
        assert train_size == (64*64*3) + 7 + 1 + self.num_options
        assert I.shape[0] == I_target.shape[0]

        o_target = np.squeeze(self.toOneHot2D(o_target, self.num_options))
        length = I.shape[0]
        #Itrain = np.reshape(I_target,(length, image_size))
        train_target = np.concatenate([q_target,g_target,o_target],axis=-1)

        return [I, q, g, oin, I_target, q_target, g_target,], [
                np.expand_dims(train_target, axis=1),
                o_target,
                value_target,
                np.expand_dims(qa, axis=1),
                np.expand_dims(ga, axis=1)]

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        tt, o1, v, qa, ga = targets
        if self.use_noise:
            noise_len = features[0].shape[0]
            z = np.random.random(size=(noise_len,self.num_hypotheses, self.noise_dim))
            return features[:4] + [z], [tt, o1, v]
        else:
            return features[:4], [tt, o1, v]

