
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

from abstract import HierarchicalAgentBasedModel
from multi_hierarchical import RobotMultiHierarchical

class RobotMultiPlanner(RobotMultiHierarchical):
    '''
    This one makes slightly different assumptions, and attempts to generate a
    set of images. Policies take us to an image goal rather than to arbitrary
    scenes.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Task def 
        '''
        super(RobotMultiPlanner, self).__init__(*args, **kwargs)
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
        option_in = Input((self._numLabels(),),name="chosen_option_in")
        prev_option_in = Input((self._numLabels(),),name="prev_option_in")
        option = Reshape([1,1,self._numLabels()])(option_in)
        option = Lambda(lambda x: K.tile(x, tile_shape))(option)

        # =====================================================================
        enc_with_option = Concatenate(
                axis=-1,
                name="add_option_info")([enc,option])

        # TODO(cpaxton): add more options here
        enc_with_option = Conv2D(self.img_num_filters,
                kernel_size=[3,3], 
                strides=(1, 1),
                padding='same')(enc_with_option)
        ins.append(option_in)
        
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
                            option=self._numLabels(),
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
                            option=self._numLabels(),
                            batchnorm=True,)

        # Predict the next joint states and gripper position. We add these back
        # in from the inputs once again, in order to make sure they don't get
        # lost in all the convolution layers above...
        x = Conv2D(self.img_num_filters/2,
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(enc_with_option)
        x = Flatten()(x)
        x = Concatenate(name="add_current_arm_info")([x, ins[1], ins[2]])
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        arm_out = Dense(arm_size,name="action_arm_goal")(x)
        gripper_out = Dense(gripper_size,name="action_gripper_goal")(x)

        # =====================================================================
        # SUPERVISOR
        # Predict the next option -- does not depend on option
        prev_option = Reshape([1,1,self._numLabels()])(prev_option_in)
        prev_option = Lambda(lambda x: K.tile(x, tile_shape))(prev_option)
        x = Concatenate(axis=-1,name="add_prev_option_to_supervisor")(
                [prev_option, enc])
        x = Conv2D(self.img_num_filters/4,
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        label_out = Dense(self._numLabels(), activation="sigmoid")(x)

        supervisor = Model(ins[:3] + [prev_option_in], [label_out])

        enc_with_option_flat = Flatten()(enc_with_option)
        decoder = Model(rep, dec, name="action_image_goal_decoder")
        next_frame_decoder = Model(
                rep2,
                dec2,
                name="action_next_image_decoder")
        features_out = [
                decoder([enc_with_option_flat,option_in]),
                arm_out,
                gripper_out,
                next_frame_decoder([enc_with_option_flat, option_in])]
        predictor = Model(ins, features_out)

        predict_goal = Model(ins, features_out[:3],)
        predict_next = Model(ins, features_out[3])

        return enc, supervisor, predictor, predict_goal, predict_next



    def train():
        '''
        Create all the models if they don't already exist
        '''
        pass
