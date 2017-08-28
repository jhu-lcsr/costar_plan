
import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from abstract import *
from multi_hierarchical import *
from robot_multi_models import *
from split import *
from mhp_loss import *

class RobotMultiPredictionSampler(RobotMultiHierarchical):

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
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(RobotMultiPredictionSampler, self).__init__(taskdef, *args, **kwargs)

        self.num_frames = 1

        self.dropout_rate = 0.5
        self.img_col_dim = 512
        self.img_num_filters = 64
        self.combined_dense_size = 128
        self.num_hypotheses = 4

        self.predictor = None
        self.train_predictor = None
        self.actor = None


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

    def _makePredictor(self, features):
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
                batchnorm=True,
                tile=True,
                option=64,#self._numLabels(),
                flatten=False,
                )


        image_outs = []
        rep, dec = GetImageDecoder(self.img_col_dim,
                            img_shape,
                            dropout_rate=self.dropout_rate,
                            kernel_size=[5,5],
                            filters=self.img_num_filters,
                            stride2_layers=3,
                            stride1_layers=0,
                            dropout=False,
                            leaky=True,
                            dense=False,
                            batchnorm=True,)

        # =====================================================================
        # Decode arm/gripper state.
        # Predict the next joint states and gripper position. We add these back
        # in from the inputs once again, in order to make sure they don't get
        # lost in all the convolution layers above...
        height4 = img_shape[0]/4
        width4 = img_shape[1]/4
        height8 = img_shape[0]/8
        width8 = img_shape[1]/8
        x = Reshape((width8,height8,self.img_num_filters))(rep)
        x = Conv2D(self.img_num_filters/2,
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(x)
        x = Flatten()(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        arm_flat = Dense(self.num_hypotheses * arm_size,name="next_arm_flat")(x)
        arm_out = Reshape((self.num_hypotheses, arm_size), name="next_arm")(arm_flat)
        gripper_flat = Dense(self.num_hypotheses * gripper_size,
                name="next_gripper_flat")(x)
        gripper_out = Reshape((self.num_hypotheses, gripper_size),
                name="next_gripper")(gripper_flat)
        #decoder = Model(rep, [dec, arm_out, gripper_out, label_out])
        decoder = Model(rep, [dec])

        # =====================================================================
        # Create many different image decoders
        for i in xrange(self.num_hypotheses):
            """
            rep, dec = GetImageDecoder(self.img_col_dim,
                                img_shape,
                                dropout_rate=self.dropout_rate,
                                kernel_size=[5,5],
                                filters=self.img_num_filters,
                                stride2_layers=3,
                                stride1_layers=0,
                                dropout=False,
                                leaky=True,
                                dense=False,
                                batchnorm=True,)
            """
            x = Conv2D(self.img_num_filters,
                    kernel_size=[5,5], 
                    strides=(1, 1),
                    padding='same')(enc)
            x = LeakyReLU(0.2)(x)

            # Create decoder
            # This maps from our latent world state back into observable images.
            #decoder = Model(rep, dec)
            decode_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="hypothesis%d"%i)(decoder(x))
            image_outs.append(decode_x)
        image_out = Concatenate(axis=1)(image_outs)

        # =====================================================================
        # Create training output. This is the flattened image, or the flattened
        # concatenation of (image + arm + gripper), which is a bit messier.
        train_out = image_out #Flatten()(image_out)

        #predictor = Model(ins, [decoder(enc), arm_out, gripper_out])
        predictor = Model(ins, [image_out])
        predictor.summary()
        train_predictor = Model(ins, train_out)

        return predictor, train_predictor

    def _fitPredictor(self, features, targets):
        if self.show_iter > 0:
            fig, axes = plt.subplots(6, 6,)
            plt.tight_layout()

        image_shape = features[0].shape[1:]
        image_size = 1.
        for dim in image_shape:
            image_size *= dim

        self.train_predictor.summary()
        self.train_predictor.compile(
                loss=MhpLoss(
                    num_hypotheses=self.num_hypotheses,
                    num_outputs=image_size),
                optimizer=self.getOptimizer())
        self.predictor.compile(loss="mse", optimizer=self.getOptimizer())

        for i in xrange(self.iter):
            idx = np.random.randint(0, features[0].shape[0], size=self.batch_size)
            x = []
            y = []
            for f in features:
                x.append(f[idx])
            for f in targets:
                y.append(np.expand_dims(f[idx],1))
                #y.append(f[idx],1))

            losses = self.train_predictor.train_on_batch(x, y)

            print "Iter %d: loss ="%(i),losses
            if self.show_iter > 0 and (i+1) % self.show_iter == 0:
                self.plotPredictions(features, targets, axes)

        self._fixWeights()

    def plotPredictions(self, features, targets, axes):
        subset = [f[range(0,60,10)] for f in features]
        data = self.predictor.predict(subset)
        #print "RESULT[0] SHAPE >>>", data[0].shape
        #print "ALL RESULTS SHAPE >>>", data.shape
        for j in xrange(6):
            jj = j * 10
            ax = axes[1][j]
            ax.imshow(np.squeeze(data[j][0]))
            ax.axis('off')
            if self.num_hypotheses > 1:
                ax = axes[2][j]
                ax.imshow(np.squeeze(data[j][1]))
                ax.axis('off')
                ax = axes[3][j]
                ax.imshow(np.squeeze(data[j][2]))
                ax.axis('off')
                ax = axes[4][j]
                ax.imshow(np.squeeze(data[j][3]))
                ax.axis('off')
            ax = axes[0][j]
            ax.imshow(np.squeeze(features[0][jj]))
            ax.axis('off')
            ax = axes[5][j]
            ax.imshow(np.squeeze(targets[0][jj]))
            ax.axis('off')

        plt.ion()
        plt.show(block=False)
        plt.pause(0.01)

    def _makeModel(self, features, arm, gripper, *args, **kwargs):
        self.predictor, self.train_predictor = \
            self._makePredictor(
                (features, arm, gripper))

    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            prev_label, goal_features, goal_arm, goal_gripper, *args, **kwargs):
        '''
        Pre-process training data.

        Then, create the model. Train based on labeled data. Remove
        unsuccessful examples.
        '''

        I = features
        q = arm
        g = gripper
        qa = arm_cmd
        ga = gripper_cmd
        oin = prev_label
        I_target = goal_features
        o_target = label

        print "sanity check:",
        print "images:", I.shape, I_target.shape
        print "joints:", q.shape,
        print "options:", oin.shape, o_target.shape

        if self.predictor is None:
            self._makeModel(I, q, g, qa, ga, oin)

        # Fit the main models
        self._fitPredictor(
                [I, q, g, oin],
                #[I_target, q_target, g_target, Inext_target])
                #[I],
                [I_target])

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
        # self._fitSupervisor([I, q, g, o_prev], o_target)
        # ===============================================
        #action_target = [qa, ga]
        #self._fitPolicies([I, q, g], action_labels, action_target)
        #self._fitBaseline([I, q, g], action_target)


