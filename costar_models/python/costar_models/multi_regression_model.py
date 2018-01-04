
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

from .abstract import AbstractAgentBasedModel
from .robot_multi_models import *
from .multi_hierarchical import *

class RobotMultiFFRegression(RobotMultiHierarchical):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        Read in taskdef for this model (or set of models). Use it to create the
        regression neural net that we can fit to compute our next action.

        Remember, here the "labels" are computed from the task model. We can
        use images and joint states together to compute next image or next
        joint state.
        '''

        super(RobotMultiFFRegression, self).__init__(taskdef, *args, **kwargs)

        self.taskdef = taskdef
        self.model = None
        
        self.dropout_rate = 0.5
        
        self.img_dense_size = 512
        self.img_col_dim = 256
        self.img_num_filters = 64
        self.robot_col_dense_size = 128
        self.robot_col_dim = 64
        self.combined_dense_size = 64
        self.pose_col_dim = 64

    def _makeModel(self, features, arm, gripper, arm_cmd, gripper_cmd, *args, **kwargs):
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
                pre_tiling_layers=1,
                post_tiling_layers=3,
                stride1_post_tiling_layers=1)

        arm_out = Dense(arm_cmd_size)(x)
        gripper_out = Dense(gripper_size)(x)

        model = Model(ins, [arm_out, gripper_out])
        optimizer = self.getOptimizer()
        model.compile(loss="mae", optimizer=optimizer)
        self.model = model

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        return [I, q, g], [np.squeeze(qa), np.squeeze(ga)]

    def trainFromGenerators(self, train_generator, test_generator, data=None, *args, **kwargs):
        [features, arm, gripper], [arm_cmd, gripper_cmd] = self._getData(**data)
        self._makeModel(features, arm, gripper, arm_cmd,
                gripper_cmd, *args, **kwargs)
        self.model.summary()
        self.model.fit_generator(
                train_generator,
                self.steps_per_epoch,
                epochs=self.epochs,
                validation_steps=self.validation_steps,
                validation_data=test_generator,)

    def predict(self, world):
        features = world.initial_features # use cached features
        if isinstance(features, list):
            assert len(features) == len(self.model.inputs)
        if self.model is None:
            raise RuntimeError('model is missing')
        features = [f.reshape((1,)+f.shape) for f in features]
        res = self.model.predict(features)
        if np.any(res[0][0] > 10.):
            plt.imshow(features[0][0])
            plt.show()

        return res
