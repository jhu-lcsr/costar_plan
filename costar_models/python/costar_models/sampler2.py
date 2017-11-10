from __future__ import print_function

import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate, Multiply
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from .abstract import *
from .callbacks import *
from .robot_multi_models import *
from .split import *
from .mhp_loss import *
from .loss import *
from .multi_sampler import *

class PredictionSampler2(RobotMultiPredictionSampler):

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(PredictionSampler2, self).__init__(taskdef, *args, **kwargs)
        self.skip_connections = False
        self.rep_size = 128
        self.PredictorCb = ImageCb

    def _makeFromHidden(self, size):
        h = Input((size,))
        ih, iw, ic = self.hidden_shape

        # ---------------------------------
        x = AddDense(h, 2*self.rep_size, "relu", self.decoder_dropout_rate)
        x = AddDense(x,int(ih*iw*ic),"relu",self.decoder_dropout_rate)
        x = Reshape((ih,iw,ic))(x)
        img = self.image_decoder(x)

        # ---------------------------------
        x = AddDense(h, 256, "relu", self.decoder_dropout_rate)
        x = AddDense(x, 512, "relu", self.decoder_dropout_rate)
        arm = AddDense(x, 6, "linear", 0.)
        gripper = AddDense(x, 1, "sigmoid", 0.)
        
        # ---------------------------------
        x = AddDense(h, 64, "relu", self.decoder_dropout_rate)
        label = AddDense(x, self.num_options, "softmax", 0.)

        model = Model(h, [img, arm, gripper, label])
        model.summary()
        self.hidden_decoder = model
        return model

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''

        # Setup feature info
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
        
        # =====================================================================
        # Load the image decoders
        img_in = Input(img_shape,name="predictor_img_in")
        encoder = self._makeImageEncoder(img_shape)
        encoder.load_weights(self._makeName(
            "pretrain_image_encoder_model",
            "image_encoder.h5f"))
        #encoder.trainable = False
        enc = encoder(img_in)
        decoder = self._makeImageDecoder(self.hidden_shape)
        decoder.load_weights(self._makeName(
            "pretrain_image_encoder_model",
            "image_decoder.h5f"))
        #decoder.trainable = False

        # =====================================================================
        # Load the arm and gripper representation
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        arm_gripper = Concatenate()([arm_in, gripper_in])
        label_in = Input((1,))
        ins = [img_in, arm_in, gripper_in, label_in]

        # =====================================================================
        # combine these models together with state information and label
        # information
        img_rep = encoder(img_in)
        dense_rep = AddDense(arm_gripper, 64, "relu", self.dropout_rate)
        img_rep = Flatten()(img_rep)
        label_rep = OneHot(self.num_options)(label_in)
        label_rep = Flatten()(label_rep)
        all_rep = Concatenate()([dense_rep, img_rep, label_rep])
        x = AddDense(all_rep, self.rep_size, "relu", self.dropout_rate)
        value_out, next_option_out = GetNextOptionAndValue(x,
                                                           self.num_options,
                                                           self.rep_size,
                                                           dropout_rate=0.5,
                                                           option_in=None)
        hidden_decoder = self._makeFromHidden(self.rep_size)
        img_x, arm_x, gripper_x, label_x = hidden_decoder(x)
        ae_outs = [img_x, arm_x, gripper_x, label_x, value_out]
        ae2 = Model(ins, ae_outs)
        ae2.compile(
            loss=["mae","mae", "mae",
                "categorical_crossentropy","binary_crossentropy"],
            optimizer=self.getOptimizer())

        #return predictor, train_predictor, None, ins, enc
        return ae2, ae2, None, ins, enc

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, q_target, g_target,] = features
        [tt, o1, v, qa, ga, I] = targets
        oin_1h = np.squeeze(self.toOneHot2D(oin, self.num_options))
        return [I, q, g, oin], [I, q, g, oin_1h, v]

    def makePredictor(self):
        # =====================================================================
        # Create many different image decoders
        image_outs = []
        arm_outs = []
        gripper_outs = []
        train_outs = []
        label_outs = []
        enc = x
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

            img_x, arm_x, gripper_x, label_x = hidden_decoder(x)

            # Create the training outputs
            train_x = Concatenate(axis=-1,name="combine_train_%d"%i)([
                            Flatten(name="flatten_img_%d"%i)(img_x),
                            arm_x,
                            gripper_x,
                            label_x])
            img_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="img_hypothesis_%d"%i)(img_x)
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

            image_outs.append(img_x)
            arm_outs.append(arm_x)
            gripper_outs.append(gripper_x)
            label_outs.append(label_x)
            train_outs.append(train_x)


        image_out = Concatenate(axis=1)(image_outs)
        arm_out = Concatenate(axis=1)(arm_outs)
        gripper_out = Concatenate(axis=1)(gripper_outs)
        label_out = Concatenate(axis=1)(label_outs)
        train_out = Concatenate(axis=1,name="all_train_outs")(train_outs)

        # =====================================================================
        # Create models to train
        if self.use_noise:
            ins += [z]
        predictor = Model(ins,
                [image_out, arm_out, gripper_out, label_out, next_option_out,
                    value_out])
        actor = None
        losses = [MhpLossWithShape(
                        num_hypotheses=self.num_hypotheses,
                        outputs=[image_size, arm_size, gripper_size, self.num_options],
                        #weights=[0.7,1.0,0.1,0.1],
                        weights=[0.3, 0.4, 0.05, 0.3],
                        loss=["mae","mae","mae","categorical_crossentropy"],
                        #stats=stats,
                        avg_weight=0.025),]
        if self.success_only and False:
            outs = [train_out, next_option_out]
            losses += ["binary_crossentropy"]
            loss_weights = [0.60, 0.40]
        else:
            outs = [train_out, next_option_out, value_out]
            loss_weights = [0.90, 0.05, 0.05]
            losses += ["categorical_crossentropy", "binary_crossentropy"]
        # =====================================================================
        # Create many different image decoders
        image_outs = []
        arm_outs = []
        gripper_outs = []
        train_outs = []
        label_outs = []
        enc = x
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

            img_x, arm_x, gripper_x, label_x = hidden_decoder(x)

            # Create the training outputs
            train_x = Concatenate(axis=-1,name="combine_train_%d"%i)([
                            Flatten(name="flatten_img_%d"%i)(img_x),
                            arm_x,
                            gripper_x,
                            label_x])
            img_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="img_hypothesis_%d"%i)(img_x)
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

            image_outs.append(img_x)
            arm_outs.append(arm_x)
            gripper_outs.append(gripper_x)
            label_outs.append(label_x)
            train_outs.append(train_x)


        image_out = Concatenate(axis=1)(image_outs)
        arm_out = Concatenate(axis=1)(arm_outs)
        gripper_out = Concatenate(axis=1)(gripper_outs)
        label_out = Concatenate(axis=1)(label_outs)
        train_out = Concatenate(axis=1,name="all_train_outs")(train_outs)

        # =====================================================================
        # Create models to train
        if self.use_noise:
            ins += [z]
        predictor = Model(ins,
                [image_out, arm_out, gripper_out, label_out, next_option_out,
                    value_out])
        actor = None
        losses = [MhpLossWithShape(
                        num_hypotheses=self.num_hypotheses,
                        outputs=[image_size, arm_size, gripper_size, self.num_options],
                        #weights=[0.7,1.0,0.1,0.1],
                        weights=[0.3, 0.4, 0.05, 0.3],
                        loss=["mae","mae","mae","categorical_crossentropy"],
                        #stats=stats,
                        avg_weight=0.025),]
        if self.success_only and False:
            outs = [train_out, next_option_out]
            losses += ["binary_crossentropy"]
            loss_weights = [0.60, 0.40]
        else:
            outs = [train_out, next_option_out, value_out]
            loss_weights = [0.90, 0.05, 0.05]
            losses += ["categorical_crossentropy", "binary_crossentropy"]

        z = Input((self.num_hypotheses, self.noise_dim))

        train_predictor = Model(ins, outs)
        train_predictor.compile(
                loss=losses,
                loss_weights=loss_weights,
                optimizer=self.getOptimizer())
        train_predictor.summary()
        predictor.compile(
                loss=["mae", "mae", "mae", "mae", "categorical_crossentropy",
                      "mae"],
                optimizer=self.getOptimizer())
