
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

        self.num_frames = 4

        self.dropout_rate = 0.5
        self.img_dense_size = 512
        self.img_col_dim = 256
        self.img_num_filters = 32
        self.robot_col_dense_size = 128
        self.robot_col_dim = 64
        self.combined_dense_size = 64
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
        '''
        images, arm, gripper = features
        arm_cmd, gripper_cmd = action
        img_shape = images.shape[1:]
        arm_size = arm.shape[-1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[-1]
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
                post_tiling_layers=2,
                time_distributed=self.num_frames)

        x = Flatten()(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)

        arm_out = Dense(arm_size)(x)
        gripper_out = Dense(gripper_size)(x)

        policy = Model(ins, [arm_out, gripper_out])
        #model = Model(img_ins, [arm_out])
        optimizer = self.getOptimizer()
        policy.compile(loss="mse", optimizer=optimizer)

        return policy

    def _makeSupervisor(self, features, label, num_labels):
        (images, arm, gripper) = features
        img_shape = images.shape[1:]
        arm_size = arm.shape[-1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[-1]
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
                post_tiling_layers=2,
                time_distributed=self.num_frames)

        x = Flatten()(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)

        label_out = Dense(num_labels, activation="sigmoid")(x)
        supervisor = Model(ins, [label_out])
        supervisor.compile(
                loss=["binary_crossentropy"],
                optimizer=self.getOptimizer())
        return x, supervisor

    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            example, reward, *args, **kwargs):
        '''
        Pre-process training data.

        Then, create the model. Train based on labeled data. Remove
        unsuccessful examples.
        '''
        action_labels = np.array([self.taskdef.index(l) for l in label])
        action_labels = np.squeeze(self.toOneHot2D(action_labels,
            len(self.taskdef.indices)))

        [features, arm, gripper, arm_cmd, gripper_cmd, action_labels], _ = \
            SplitIntoChunks(
                datasets=[features, arm, gripper, arm_cmd, gripper_cmd,
                    action_labels],
                reward=None, reward_threshold=0.,
                labels=example,
                chunk_length=self.num_frames,
                step_size=self.partition_step_size,
                front_padding=True,
                rear_padding=False,
                start_off=1,
                end_off=1)

        # image inputs
        I = features[:,1:-1]
        q = arm[:,1:-1]
        g = gripper[:,1:-1]
        print I.shape
        print q.shape

        for i in xrange(self.num_frames):
            for j in xrange(self.num_frames):
                plt.subplot(i+1,j+1,self.num_frames**2)
                plt.imshow(I[i,j])
        plt.show()

        self._makeModel(features, arm, gripper, arm_cmd,
                gripper_cmd, action_labels, *args, **kwargs)

        label_target = np.squeeze(action_labels[:,-1,:])
        arm_target = np.squeeze(arm_cmd[:,-1,:])
        gripper_target = np.squeeze(arm_cmd[:,-1,:])

        action_target = [arm_target, gripper_target]

        self._fitSupervisor([features, arm, gripper], action_labels,
                label_target)
        self._fitPolicies([features, arm, gripper], action_labels, action_target)
        self._fitBaseline([features, arm, gripper], action_target)


    def save(self):
        '''
        Store the model to disk here.
        '''
        pass
