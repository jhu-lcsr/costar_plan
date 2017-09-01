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
from .preprocess import *
from .robot_multi_models import *
from .split import *

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

        self.num_frames = 1

        self.dropout_rate = 0.5
        self.img_dense_size = 1024
        self.img_col_dim = 512
        self.img_num_filters = 128
        self.combined_dense_size = 128
        self.partition_step_size = 2

    def _makeModel(self, features, arm, gripper, arm_cmd, gripper_cmd, label, *args, **kwargs):
        self._makeHierarchicalModel(
                (features, arm, gripper),
                (arm_cmd, gripper_cmd),
                label)

    def _makePolicy(self, features, action, hidden=None):
        '''
        We need to use the task definition to create our high-level model, and
        we need to use our data to initialize the low level models that will be
        predicting our individual actions.

        Parameters:
        -----------
        features: input list of features representing current state. Note that
                  this is included for completeness in the hierarchical model,
                  but is not currently used in this implementation (and ideally
                  would not be).
        action: input list of action outputs (arm and gripper commands for the
                robot tasks).
        hidden: "hidden" embedding of latent world state (input)
        '''
        images, arm, gripper = features
        arm_cmd, gripper_cmd = action
        img_shape = images.shape[1:]
        arm_size = arm.shape[-1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[-1]
        else:
            gripper_size = 1
        

        x = Conv2D(self.img_num_filters/4,
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(hidden)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)

        arm_out = Dense(arm_size)(x)
        gripper_out = Dense(gripper_size)(x)

        policy = Model(self.supervisor.inputs[:3], [arm_out, gripper_out])

        return policy

    def _makeSupervisor(self, features):
        (images, arm, gripper) = features
        img_shape = images.shape[1:]
        arm_size = arm.shape[-1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[-1]
        else:
            gripper_size = 1

        ins, enc = GetEncoder(img_shape,
                arm_size,
                gripper_size,
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                leaky=False,
                dropout=False,
                pre_tiling_layers=0,
                post_tiling_layers=3,
                kernel_size=[5,5],
                dense=False,
                tile=True,
                option=None,#self._numLabels(),
                flatten=False,
                )

        # Tile on the option -- this is where our transition model comes in.
        # Options are represented as a one-hot vector added to all possible
        # positions in the image, and essentially give us _numLabels()
        # additional image channels.
        tile_width = img_shape[0]/(2**3)
        tile_height = img_shape[1]/(2**3)
        tile_shape = (1, tile_width, tile_height, 1)

        rep, dec = GetDecoder(self.img_col_dim,
                            img_shape,
                            arm_size,
                            gripper_size,
                            dropout_rate=self.dropout_rate,
                            kernel_size=[5,5],
                            filters=self.img_num_filters,
                            stride2_layers=3,
                            stride1_layers=0,
                            dropout=False,
                            leaky=True,
                            dense=False,
                            option=None,
                            batchnorm=True,)
        rep2, dec2 = GetDecoder(self.img_col_dim,
                            img_shape,
                            arm_size,
                            gripper_size,
                            dropout_rate=self.dropout_rate,
                            kernel_size=[5,5],
                            filters=self.img_num_filters,
                            stride2_layers=3,
                            stride1_layers=0,
                            dropout=False,
                            leaky=True,
                            dense=False,
                            option=None,
                            batchnorm=True,)


        # =====================================================================
        # SUPERVISOR
        # Predict the next option -- does not depend on option
        prev_option_in = Input((1,),name="prev_option_in")
        prev_option = OneHot(size=64)(prev_option_in)
        prev_option = Reshape([1,1,64])(prev_option)
        prev_option = Lambda(lambda x: K.tile(x, tile_shape))(prev_option)
        x = Concatenate(axis=-1,name="add_prev_option_to_supervisor")(
                [prev_option, enc])
        for _ in xrange(2):
            # Repeat twice to scale down to a very small size -- this will help
            # a little with the final image layers
            x = Conv2D(self.img_num_filters/4,
                    kernel_size=[5, 5], 
                    strides=(2, 2),
                    padding='same')(x)
            x = Dropout(self.dropout_rate)(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        label_out = Dense(64, activation="sigmoid",name="next_option")(x)
        decoder = Model(rep, dec, name="image_decoder")
        decoder2 = Model(rep2, dec2, name="next_frame_image_decoder")

        # =====================================================================
        # Add in the chosen option
        # ---------------------------------------------------------------------
        option = Reshape([1,1,64])(label_out)
        option = Lambda(lambda x: K.tile(x, tile_shape))(option)
        enc_with_option = Concatenate(
                axis=-1,
                name="add_option_info")([enc,option])
        goal_enc_with_option = Conv2D(self.img_num_filters,
                kernel_size=[5,5], 
                strides=(1, 1),
                padding='same')(enc_with_option)
        goal_enc_with_option = LeakyReLU(0.2,
                name='goal_encoding_with_option')(goal_enc_with_option)
        # ---------------------------------------------------------------------
        next_frame_enc_with_option = Conv2D(self.img_num_filters,
                kernel_size=[5,5], 
                strides=(1, 1),
                padding='same')(enc_with_option)
        next_frame_enc_with_option = LeakyReLU(0.2,
                name='next_frame_encoding_with_option')(next_frame_enc_with_option)
        
        # Predict the next joint states and gripper position. We add these back
        # in from the inputs once again, in order to make sure they don't get
        # lost in all the convolution layers above...
        x = Conv2D(self.img_num_filters/2,
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(goal_enc_with_option)
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Concatenate(name="add_current_arm_info")([x, ins[1], ins[2]])
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        arm_out = Dense(arm_size,name="action_arm_goal")(x)
        gripper_out = Dense(gripper_size,name="action_gripper_goal")(x)

        # =====================================================================
        # PREDICTOR AND LATENT STATE MODEL
        # Create the necessary models
        goal_enc_with_option_flat = Flatten(name="goal_flat")(goal_enc_with_option)
        next_frame_enc_with_option_flat = Flatten(name="next_frame_flat")(next_frame_enc_with_option)

        features_out = [
                decoder([goal_enc_with_option_flat]),
                arm_out,
                gripper_out,
                #decoder2([next_frame_enc_with_option_flat])]

        supervisor = Model(ins + [prev_option_in], [label_out])
        predictor = Model(ins + [prev_option_in], features_out + [label_out])
        predict_goal = Model(ins + [prev_option_in], features_out[:3],)
        predict_next = Model(ins + [prev_option_in], features_out[3])

        return enc, supervisor, predictor, predict_goal, predict_next

    def _fitPredictor(self, features, targets):
        if self.show_iter > 0:
            fig, axes = plt.subplots(5, 5,)

        self._unfixWeights()
        self.predictor.compile(
                loss=(["mse"]*4+["binary_crossentropy"]),
                optimizer=self.getOptimizer())
        self.predictor.summary()

        for i in xrange(self.iter):
            idx = np.random.randint(0, features[0].shape[0], size=self.batch_size)
            x = []
            y = []
            for f in features:
                x.append(f[idx])
            for f in targets:
                y.append(f[idx])

            losses = self.predictor.train_on_batch(x, y)

            print("Iter %d: loss ="%(i),losses)
            if self.show_iter > 0 and (i+1) % self.show_iter == 0:
                self.plotInfo(features, targets, axes)

        self._fixWeights()

    def plotInfo(self, features, targets, axes):
        # debugging: plot every 5th image from the dataset
        subset = [f[range(0,25,5)] for f in features]
        data = self.predictor.predict(subset)
        for j in xrange(5):
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

    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            next_features, next_arm, next_gripper,
            prev_label, goal_features, goal_arm, goal_gripper, *args, **kwargs):
        '''
        Pre-process training data.
        
        Then, create the model. Train based on labeled data. Remove
        unsuccessful examples.
        '''

        # ================================================
        # Set up variable names -- just to make things a bit cleaner
        I = features
        q = arm
        g = gripper
        qa = arm_cmd
        ga = gripper_cmd
        oin = prev_label
        I_target = goal_features
        Inext_target = next_features
        o_target = label
        q_target = goal_arm
        g_target = goal_gripper
        action_labels = label

        if self.supervisor is None:
            self._makeModel(I, q, g, qa, ga, oin)

        # Fit the main models
        self._fitPredictor(
                [I, q, g, prev_label],
                [I_target, q_target, g_target, Inext_target,
                    to_categorical(o_target, 64)])

        # ===============================================
        # Might be useful if you start getting shitty results... one problem we
        # observed was accidentally training the embedding weights when
        # learning all your policies.
        #fig, axes = plt.subplots(5, 5,)
        #self.plotInfo(
        #        [I, q, g, oin],
        #        [I_target, q_target, g_target, Inext_target],
        #        axes,
        #        )
        #self._fitSupervisor([I, q, g, o_prev], o_target)
        # ===============================================

        action_target = [qa, ga]
        #self._fitPolicies([I, q, g], action_labels, action_target)
        #self._fitBaseline([I, q, g], action_target)


