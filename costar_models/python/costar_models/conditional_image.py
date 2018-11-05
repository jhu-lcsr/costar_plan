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

from .callbacks import *
from .multi_sampler import *
from .data_utils import GetNextGoal, ToOneHot
from .multi import *
from .loss import *

class ConditionalImage(RobotMultiPredictionSampler):
    '''
    Version of the sampler that only produces results conditioned on a
    particular action; this version does not bother trying to learn a separate
    distribution for each possible state.

    This one generates:
      - image
      - arm command
      - gripper command
    '''

    def __init__(self, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.

        Parameters:
        -----------
        taskdef: definition of the problem used to create a task model
        '''
        super(ConditionalImage, self).__init__(*args, **kwargs)
        self.PredictorCb = ImageWithFirstCb
        self.rep_size = 256
        self.num_transforms = 3
        self.transform_model = None
        self.save_encoder_decoder = self.retrain
        self.encoder_channels = 8

        if self.use_noise:
            raise NotImplementedError('noise vectors not supported for'
                                      'conditional_image model')

    def _makePredictor(self, features):
        # =====================================================================
        # Create many different image decoders
        (images, arm, gripper) = features
        img_shape, image_size, arm_size, gripper_size = self._sizes(
                images,
                arm,
                gripper)

        # =====================================================================
        # Load the image decoders
        img_in = Input(img_shape,name="predictor_img_in")
        img0_in = Input(img_shape,name="predictor_img0_in")
        #arm_in = Input((arm_size,))
        #gripper_in = Input((gripper_size,))
        #arm_gripper = Concatenate()([arm_in, gripper_in])
        label_in = Input((1,))
        ins = [img0_in, img_in]

        encoder = MakeImageEncoder(self, img_shape)
        decoder = MakeImageDecoder(self, self.hidden_shape)

        LoadEncoderWeights(self, encoder, decoder)

        # =====================================================================
        # Load the arm and gripper representation
        h = encoder([img0_in, img_in])

        if self.validate:
            self.loadValidationModels(arm_size, gripper_size, h0, h)

        next_option_in = Input((1,), name="next_option_in")
        next_option_in2 = Input((1,), name="next_option_in2")
        ins += [next_option_in, next_option_in2]

        # =====================================================================
        # Apply transforms
        y = Flatten()(OneHot(self.num_options)(next_option_in))
        y2 = Flatten()(OneHot(self.num_options)(next_option_in2))

        tform = self._makeTransform() if not self.dense_transform else self._makeDenseTransform()
        x = tform([h,y])
        x2 = tform([x,y2])

        image_out, image_out2 = decoder([x]), decoder([x2])

        # Compute classifier on the last transform
        if not self.no_disc:
            image_discriminator = LoadGoalClassifierWeights(self,
                    make_classifier_fn=MakeImageClassifier,
                    img_shape=img_shape)
            #disc_out1 = image_discriminator([img0_in, image_out])
            disc_out2 = image_discriminator([img0_in, image_out2])

        # Create custom encoder loss
        if self.enc_loss:
            loss = EncoderLoss(self.image_encoder, self.loss)
            enc_losses = [loss, loss]
            enc_outs = [x, x2]
            enc_wts = [1e-2, 1e-2]
            img_loss_wt = 1.
        else:
            enc_losses = []
            enc_outs = []
            enc_wts = []
            img_loss_wt = 1.

        # Create models to train
        if self.no_disc:
            disc_wt = 0.
        else:
            disc_wt = 1e-3
        if self.no_disc:
            train_predictor = Model(ins + [label_in],
                    [image_out, image_out2] + enc_outs)
            train_predictor.compile(
                    loss=[self.loss, self.loss,] + enc_losses,
                    loss_weights=[img_loss_wt, img_loss_wt] + enc_wts,
                    optimizer=self.getOptimizer())
        else:
            train_predictor = Model(ins + [label_in],
                    #[image_out, image_out2, disc_out1, disc_out2] + enc_outs)
                    [image_out, image_out2, disc_out2] + enc_outs)
            train_predictor.compile(
                    loss=[self.loss, self.loss, "categorical_crossentropy"] + enc_losses,
                    #loss_weights=[img_loss_wt, img_loss_wt, 0.9*disc_wt, disc_wt] + enc_wts,
                    loss_weights=[img_loss_wt, img_loss_wt, disc_wt] + enc_wts,
                    optimizer=self.getOptimizer())
        train_predictor.summary()
        return None, train_predictor, None, ins, h

    def _getData(self, *args, **kwargs):
        features, targets = GetAllMultiData(self.num_options, *args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        I_target2, o2 = GetNextGoal(I_target, o1)
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        oin_1h = ToOneHot(oin, self.num_options)
        o1_1h = ToOneHot(o1, self.num_options)
        o2_1h = ToOneHot(o2, self.num_options)
        qa = np.squeeze(qa)
        ga = np.squeeze(ga)
        features = [I0, I, o1, o2, oin]
        if self.validate:
            # Return the specific set of features that are just for validation
            return (features,
                    [I_target, I_target2, o1_1h, v, qa, ga, o2_1h])
        elif self.no_disc:
            targets = [I_target, I_target2]
        else:
            # Uncomment if you want to try the whole "two discriminator" thing
            # again -- this might need a more fully supported option
            #targets = [I_target, I_target2, o1_1h, o2_1h]
            targets = [I_target, I_target2, o2_1h]
        if self.enc_loss:
            targets += [I_target, I_target2]
        return features, targets


    def loadValidationModels(self, arm_size, gripper_size, h0, h):

        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        arm_gripper = Concatenate()([arm_in, gripper_in])
        label_in = Input((1,))

        print(">>> GOAL_CLASSIFIER")
        image_discriminator = LoadGoalClassifierWeights(self,
                    make_classifier_fn=MakeImageClassifier,
                    img_shape=(64, 64, 3))
        image_discriminator.compile(loss="categorical_crossentropy",
                                    metrics=["accuracy"],
                                    optimizer=self.getOptimizer())
        self.discriminator = image_discriminator

        print(">>> VALUE MODEL")
        self.value_model = GetValueModel(h, self.num_options, 128,
                self.decoder_dropout_rate)
        self.value_model.compile(loss="mae", optimizer=self.getOptimizer())
        self.value_model.load_weights(self.makeName("secondary", "value"))

        print(">>> NEXT MODEL")
        self.next_model = GetNextModel(h, self.num_options, 128,
                self.decoder_dropout_rate)
        self.next_model.compile(loss="mae", optimizer=self.getOptimizer())
        self.next_model.load_weights(self.makeName("secondary", "next"))

        print(">>> ACTOR MODEL")
        self.actor = GetActorModel(h, self.num_options, arm_size, gripper_size,
                self.decoder_dropout_rate)
        self.actor.compile(loss="mae",optimizer=self.getOptimizer())
        self.actor.load_weights(self.makeName("secondary", "actor"))

        print(">>> POSE MODEL")
        self.pose_model = GetPoseModel(h, self.num_options, arm_size, gripper_size,
                self.decoder_dropout_rate)
        self.pose_model.compile(loss="mae",optimizer=self.getOptimizer())
        self.pose_model.load_weights(self.makeName("secondary", "pose"))

        print(">>> Q MODEL")
        self.q_model = GetNextModel(h, self.num_options, 128,
                self.decoder_dropout_rate)
        self.q_model.compile(loss="mae", optimizer=self.getOptimizer())
        self.q_model.load_weights(self.makeName("secondary", "q"))

    def pnext(self, hidden0, hidden, prev_option):
        '''
        Visualize based on hidden
        '''
        p, done = self.next_model.predict([hidden0, hidden, prev_option])
        return p, done

    def q(self, hidden0, hidden, prev_option):
        #p, done, value = self.q_model.predict([hidden0, hidden, prev_option])
        p, done = self.q_model.predict([hidden0, hidden, prev_option])
        return p, done

    def value(self, hidden, *args, **kwargs):
        v = self.value_model.predict([hidden])
        return v

    def act(self, *args, **kwargs):
        raise NotImplementedError('act() not implemented')

