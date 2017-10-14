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

class RobotMultiImageSampler(RobotMultiPredictionSampler):

    '''
    Image-only version of the prediction sampler. This just looks at whether or
    not we can predict images using the MHP loss and does not look at secondary
    problems, like whether or not we can predict grasp poses, or learning the
    actor network for joint states.

    Results generally show this one converging much faster -- but not
    necessarily to results that are as useful.

    This class is set up as a SUPERVISED learning problem -- for more
    interactive training we will need to add data from an appropriate agent.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(RobotMultiImageSampler, self).__init__(taskdef, *args, **kwargs)
        self.num_features = 4
        self.num_hypotheses = 4
        self.steps_down = 2
        self.steps_up = 4
        self.steps_up_no_skip = 2
        self.encoder_stride1_steps = 2

        self.PredictorCb = PredictorShowImageOnly

        # ===================================================================
        # These are hard coded settings -- tweaking them may break a bunch of
        # things.
        self.use_prev_option = True
        self.always_same_transform = False

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
                [arm_size, gripper_size],
                self.img_col_dim,
                dropout_rate=self.dropout_rate,
                filters=self.img_num_filters,
                leaky=False,
                dropout=True,
                pre_tiling_layers=self.extra_layers,
                post_tiling_layers=self.steps_down,
                stride1_post_tiling_layers=self.encoder_stride1_steps,
                pose_col_dim=self.pose_col_dim,
                kernel_size=[5,5],
                dense=self.dense_representation,
                batchnorm=True,
                tile=True,
                flatten=False,
                use_spatial_softmax=True,
                option=self.num_options,
                output_filters=self.tform_filters,
                )
        img_in, arm_in, gripper_in, option_in = ins
        if self.use_noise:
            z = Input((self.num_hypotheses, self.noise_dim))

        # =====================================================================
        # Create the decoders for image
        if self.skip_connections:
            skips.reverse()
        self.image_decoder = self._makeImageDecoder(img_shape, [3,3], skips)

        # =====================================================================
        # Create many different image decoders
        image_outs = []
        stats = []
        if self.always_same_transform:
            transform = self._getTransform(0)
        for i in range(self.num_hypotheses):
            if not self.always_same_transform:
                transform = self._getTransform(i)

            if i == 0:
                transform.summary()
            if self.use_noise:
                zi = Lambda(lambda x: x[:,i], name="slice_z%d"%i)(z)
                x = transform([enc, zi])
            else:
                x = transform([enc])

            if self.sampling:
                x, mu, sigma = x
                stats.append((mu, sigma))
            
            # This maps from our latent world state back into observable images.
            if self.skip_connections:
                decoder_inputs = [x] + skips
            else:
                decoder_inputs = [x]

            img_x = self.image_decoder(decoder_inputs)

            img_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="img_hypothesis_%d"%i)(img_x)

            image_outs.append(img_x)

        image_out = Concatenate(axis=1)(image_outs)

        # =====================================================================
        # Create models to train
        if self.use_noise:
            ins += [z]
        predictor = Model(ins ,
                [image_out])
        actor = None
        train_predictor = Model(ins,
                [image_out])

        # =====================================================================
        # Create models to train
        train_predictor.compile(
                loss=[
                    MhpLossWithShape(
                        num_hypotheses=self.num_hypotheses,
                        outputs=[image_size],
                        weights=[1.0],
                        loss=["mae"],
                        avg_weight=0.05,
                        stats=stats
                        )],
                optimizer=self.getOptimizer())
        predictor.compile(loss="mae", optimizer=self.getOptimizer())
        train_predictor.summary()

        return predictor, train_predictor, actor

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        tt, o1, v, qa, ga, I = targets
        if self.use_noise:
            noise_len = features[0].shape[0]
            z = np.random.random(size=(noise_len,self.num_hypotheses,self.noise_dim))
            return (features[:self.num_features] + [z],
                    [np.expand_dims(I,axis=1)])
        else:
            return (features[:self.num_features],
                    [np.expand_dims(I,axis=1)])
