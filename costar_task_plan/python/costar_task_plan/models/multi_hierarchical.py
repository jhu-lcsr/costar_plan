
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

from robot_multi_models import *
from split import *

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
        super(RobotMultiHierarchical, self).__init__(*args, **kwargs)
        self.taskdef = taskdef

        self.num_frames = 10

        self.dropout_rate = 0.5
        self.img_dense_size = 512
        self.img_col_dim = 256
        self.img_num_filters = 32
        self.robot_col_dense_size = 128
        self.robot_col_dim = 64
        self.combined_dense_size = 64
        self.partition_step_size = 2

    def _makeModel(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            example, *args, **kwargs):
        '''
        We need to use the task definition to create our high-level model, and
        we need to use our data to initialize the low level models that will be
        predicting our individual actions.
        '''
        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1

        ins, x = GetEncoderConvLSTM(
                img_shape,
                arm_size,
                gripper_size,
                dropout_rate=self.dropout_rate,
                filters=self.img_num_filters,
                tile=True,
                pre_tiling_layers=1,
                post_tiling_layers=2)

        arm_out = Dense(arm_size)(x)
        gripper_out = Dense(gripper_size)(x)

        model = Model(ins, [arm_out, gripper_out])
        #model = Model(img_ins, [arm_out])
        optimizer = self.getOptimizer()
        model.compile(loss="mse", optimizer=optimizer)
        self.model = model

    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            example, reward, *args, **kwargs):
        '''
        Pre-process training data.

        Then, create the model. Train based on labeled data. Remove
        unsuccessful examples.
        '''
        print label
        action_labels = np.array([self.taskdef.index(l) for l in label])

        [features, arm, gripper, arm_cmd, gripper_cmd, actions], _ = \
            SplitIntoChunks(
                datasets=[features, arm, gripper, arm_cmd, gripper_cmd,
                    action_labels],
                reward=None, reward_threshold=0.,
                labels=example,
                chunk_length=self.num_frames,
                step_size=self.partition_step_size,
                front_padding=True,
                rear_padding=False,)

        self._makeModel(features, arm, gripper, arm_cmd,
                gripper_cmd, actions,
                example, *args, **kwargs)
        self.model.summary()

    def save(self):
        '''
        Store the model to disk here.
        '''
        pass
