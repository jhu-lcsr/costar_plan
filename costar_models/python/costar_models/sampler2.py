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
        self.rep_size = 128
        self.encoder_channels = 128
        self.PredictorCb = ImageCb
        self.skip_shape = (64,64,32)

    def _makeToHidden(self, img_shape, arm_size, gripper_size, rep_size):
        img0_in = Input(img_shape,name="predictor_img0_in")
        img_in = Input(img_shape,name="predictor_img_in")
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        label_in = Input((1,))
        ins = [img0_in, img_in, arm_in, gripper_in, label_in]

        if self.skip_connections:
            img_rep, skip_rep = self.image_encoder([img0_in, img_in])
        else:
            #img_rep = self.image_encoder(img_in)
            img_rep = self.image_encoder([img0_in, img_in])
        state_rep = self.state_encoder([arm_in, gripper_in, label_in])
        # Compress the size of the network
        x = AddConv2D(img_rep, 32, [1,1], 1, self.dropout_rate, "same")
        x = Flatten()(x)
        x = Concatenate()([x, state_rep])
        x = AddDense(x, self.rep_size, "relu", self.dropout_rate)

        if self.skip_connections:
            model = Model(ins, [x, skip_rep], name="encode_hidden_state")
        else:
            model = Model(ins, x, name="encode_hidden_state")
        model.compile(loss="mae", optimizer=self.getOptimizer())
        self.hidden_encoder = model
        return model

    def _makeFromHidden(self, size):
        h = Input((size,))
        ih, iw, ic = self.hidden_shape

        # ---------------------------------
        x = h
        x = AddDense(x,int(ih*iw*ic),"lrelu",self.decoder_dropout_rate)
        x = Reshape((ih,iw,ic))(x)
        if self.skip_connections:
            skip_in = Input(self.skip_shape)
            ins = [x, skip_in]
            hidden_decoder_ins = [h, skip_in]
        else:
            ins = x
            hidden_decoder_ins = h
        img = self.image_decoder(ins)

        # ---------------------------------
        x = AddDense(h, 256, "lrelu", self.decoder_dropout_rate)
        x = AddDense(x, 512, "lrelu", self.decoder_dropout_rate)
        arm = AddDense(x, 6, "linear", 0.)
        gripper = AddDense(x, 1, "sigmoid", 0.)
        
        # ---------------------------------
        x = AddDense(h, 64, "lrelu", self.decoder_dropout_rate)
        label = AddDense(x, self.num_options, "softmax", 0.)

        model = Model(hidden_decoder_ins, [img, arm, gripper, label])
        model.summary()
        self.hidden_decoder = model
        return model

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''

        (images, arm, gripper) = features
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                images,
                arm,
                gripper)

        # =====================================================================
        # Load the image decoders
        img_in = Input(img_shape,name="predictor_img_in")
        img0_in = Input(img_shape,name="predictor_img0_in")
        encoder = self._makeImageEncoder(img_shape)
        encoder.load_weights(self._makeName(
            "pretrain_image_encoder_model",
            "image_encoder.h5f"))
        encoder.trainable = False
        enc = encoder([img0_in, img_in])
        if self.skip_connections:
            decoder = self._makeImageDecoder(self.hidden_shape,self.skip_shape)
        else:
            decoder = self._makeImageDecoder(self.hidden_shape)
        encoder.summary()
        decoder.summary()
        decoder.load_weights(self._makeName(
            "pretrain_image_encoder_model",
            "image_decoder.h5f"))
        decoder.trainable = False


        sencoder = self._makeStateEncoder(arm_size, gripper_size, False)
        sencoder.load_weights(self._makeName(
            "pretrain_state_encoder_model", "state_encoder.h5f"))
        sdecoder = self._makeStateDecoder(arm_size, gripper_size)
        sdecoder.load_weights(self._makeName(
            "pretrain_state_encoder_model", "state_decoder.h5f"))

        # =====================================================================
        # Load the arm and gripper representation
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        arm_gripper = Concatenate()([arm_in, gripper_in])
        label_in = Input((1,))
        ins = [img0_in, img_in, arm_in, gripper_in, label_in]

        # =====================================================================
        # combine these models together with state information and label
        # information
        hidden_encoder = self._makeToHidden(img_shape, arm_size, gripper_size, self.rep_size)
        if self.skip_connections:
            h, skip_rep = hidden_encoder(ins)
        else:
            h = hidden_encoder(ins)
        value_out, next_option_out = GetNextOptionAndValue(h,
                                                           self.num_options,
                                                           self.rep_size,
                                                           dropout_rate=0.5,
                                                           option_in=None)
        hidden_decoder = self._makeFromHidden(self.rep_size)
        if self.skip_connections:
            #img_x = hidden_decoder([x, skip_rep])
            img_x, arm_x, gripper_x, label_x = hidden_decoder([h, skip_rep])
        else:
            #img_x = hidden_decoder(x)
            img_x, arm_x, gripper_x, label_x = hidden_decoder(h)
        #ae_outs = [img_x, arm_x, gripper_x, label_x, ] #value_out]
        ae_outs = [img_x, arm_x, gripper_x, label_x]
        ae2 = Model(ins, ae_outs)
        ae2.compile(
            loss=["mae","mae", "mae",
                "categorical_crossentropy",],
            loss_weights=[1.,1.,.2,0.1,],#0.25],
            optimizer=self.getOptimizer())

        #return predictor, train_predictor, None, ins, enc
        return ae2, ae2, None, ins, enc

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        [I, q, g, oin, q_target, g_target,] = features
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        [tt, o1, v, qa, ga, I] = targets
        oin_1h = np.squeeze(self.toOneHot2D(oin, self.num_options))
        return [I0, I, q, g, oin], [I, q, g, oin_1h]

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
