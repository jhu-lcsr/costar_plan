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
from keras.layers.merge import Concatenate
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

class RobotMultiPredictionSampler(RobotMultiHierarchical):

    '''
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
        self.tform_filters = 64
        self.combined_dense_size = 128
        self.num_hypotheses = 4
        self.num_transforms = 3

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
                option=64,
                flatten=False,
                output_filters=self.tform_filters,
                )
        gins, genc = GetEncoder(img_shape,
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
                #option=64,
                flatten=False,
                output_filters=self.tform_filters,
                )


        image_outs = []
        arm_outs = []
        gripper_outs = []
        train_outs = []

        rep, dec = GetImageDecoder(self.img_col_dim,
                            img_shape,
                            dropout_rate=self.dropout_rate,
                            kernel_size=[5,5],
                            filters=self.img_num_filters,
                            stride2_layers=3,
                            stride1_layers=0,
                            tform_filters=self.tform_filters,
                            dropout=False,
                            leaky=True,
                            dense=False,
                            batchnorm=True,)

        # =====================================================================
        # Decode arm/gripper state.
        # Predict the next joint states and gripper position. We add these back
        # in from the inputs once again, in order to make sure they don't get
        # lost in all the convolution layers above...
        height4 = int(img_shape[0]/4)
        width4 = int(img_shape[1]/4)
        height8 = int(img_shape[0]/8)
        width8 = int(img_shape[1]/8)
        x = Reshape((width8,height8,self.tform_filters))(rep)
        x = Conv2D(int(self.img_num_filters/2),
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(x)
        x = Flatten()(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        arm_out_x = Dense(arm_size,name="next_arm")(x)
        gripper_out_x = Dense(gripper_size,
                name="next_gripper_flat")(x)

        decoder = Model(rep, [dec, arm_out_x, gripper_out_x], name="decoder")
        aimg_out, aarm_out, agripper_out = decoder(enc)

        # =====================================================================
        # Create many different image decoders
        decoder.trainable = False
        enc.trainable = False
        for i in range(self.num_hypotheses):
            x = enc
            for j in range(self.num_transforms):
                x = Conv2D(self.tform_filters,
                        kernel_size=[5,5], 
                        strides=(1, 1),
                        padding='same',
                        name="transform_%d_%d"%(i,j))(x)
                x = BatchNormalization(momentum=0.9,
                                      name="normalize_%d_%d"%(i,j))(x)
                x = LeakyReLU(0.2,name="lrelu_%d_%d"%(i,j))(x)
            
            # This maps from our latent world state back into observable images.
            #decoder = Model(rep, dec)
            img_x, arm_x, gripper_x = decoder(x)
            img_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="img_hypothesis_%d"%i)(img_x)
            arm_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="arm_hypothesis_%d"%i)(arm_x)
            gripper_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="gripper_hypothesis_%d"%i)(gripper_x)

            # Create the training outputs
            train_x = Concatenate(axis=-1,name="combine_train_%d"%i)([
                            Flatten(name="flatten_img_%d"%i)(img_x),
                            Flatten(name="flatten_arm_%d"%i)(arm_x),
                            Flatten(name="flatten_gripper_%d"%i)(gripper_x)])
            train_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="flattened_hypothesis_%d"%i)(train_x)

            image_outs.append(img_x)
            arm_outs.append(arm_x)
            gripper_outs.append(gripper_x)
            train_outs.append(train_x)

        image_out = Concatenate(axis=1)(image_outs)
        arm_out = Concatenate(axis=1)(arm_outs)
        gripper_out = Concatenate(axis=1)(gripper_outs)
        train_out = Concatenate(axis=1)(train_outs)

        # =====================================================================
        # Training the actor policy
        y = Concatenate(axis=-1,name="combine_goal_current")([enc, genc])
        y = Conv2D(int(self.img_num_filters/4),
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(y)
        y = Dropout(self.dropout_rate)(y)
        y = LeakyReLU(0.2)(y)
        y = Flatten()(y)
        y = Dense(self.combined_dense_size)(y)
        y = Dropout(self.dropout_rate)(y)
        y = LeakyReLU(0.2)(y)
        #arm_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="arm_action")(
        #        Dense(arm_size)(y))
        #gripper_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="gripper_action")(
        #        Dense(gripper_size)(y))
        arm_cmd_out = Dense(arm_size,name="arm_action")(y)
        gripper_cmd_out = Dense(gripper_size,name="gripper_action")(y)


        # =====================================================================
        # Create models to train

        #predictor = Model(ins, [decoder(enc), arm_out, gripper_out])
        predictor = Model(ins, [image_out, arm_out, gripper_out])
        actor = Model(ins + gins, [arm_out, gripper_out])
        train_predictor = Model(ins + gins, [aimg_out,
                                            aarm_out,
                                            agripper_out,
                                            train_out,
                                             arm_cmd_out,
                                             gripper_cmd_out,])

        return predictor, train_predictor, actor

    def _fitPredictor(self, features, targets,):
        if self.show_iter > 0:
            fig, axes = plt.subplots(6, 6,)
            plt.tight_layout()

        image_shape = features[0].shape[1:]
        image_size = 1.
        for dim in image_shape:
            image_size *= dim

        TRAIN_TARGET = 3
        for i in range(features[0].shape[0]):
            img1 = targets[TRAIN_TARGET][i,0,:int(image_size)].reshape((64,64,3))
            img2 = features[4][i]
            if not np.all(img1 == img2):
                print(i,"failed")
                plt.subplot(1,2,1); plt.imshow(img1);
                plt.subplot(1,2,2); plt.imshow(img2);
                plt.show()

        if self.show_iter == 0 or self.show_iter == None:
            modelCheckpointCb = ModelCheckpoint(
                filepath=self.name+"_predictor_weights.h5f",
                verbose=1,
                save_best_only=False # does not work without validation wts
            )
            imageCb = PredictorShowImage(
                self.predictor,
                features=features[:4],
                targets=targets[TRAIN_TARGET:],
                num_hypotheses=self.num_hypotheses,
                verbose=True,
                min_idx=0,
                max_idx=66,
                step=11,)
            self.train_predictor.fit(features,
                    #[np.expand_dims(f,1) for f in targets],
                    targets,
                    callbacks=[modelCheckpointCb, imageCb],
                    epochs=self.epochs)
        else:
            for i in range(self.iter):
                idx = np.random.randint(0, features[0].shape[0], size=self.batch_size)
                x = []
                y = []
                for f in features:
                    x.append(f[idx])
                for f in targets:
                    y.append(np.expand_dims(f[idx],1))
                yimg = y[TRAIN_TARGET][:,0,:int(image_size)]
                yimg = yimg.reshape((self.batch_size,64,64,3))
                for j in xrange(self.batch_size):
                    if not np.all(x[4][j] == yimg[j]):
                        plt.subplot(1,3,1); plt.imshow(x[0][j]);
                        plt.subplot(1,3,2); plt.imshow(x[4][j]);
                        plt.subplot(1,3,3); plt.imshow(yimg[j]);
                        plt.show()
        
                losses = self.train_predictor.train_on_batch(x, y)

                print("Iter %d: loss ="%(i),losses)
                if self.show_iter > 0 and (i+1) % self.show_iter == 0:
                    self.plotPredictions(features, targets, axes)

        self._fixWeights()

    def plotPredictions(self, features, targets, axes):
        STEP = 20
        idxs = range(0,120,STEP)
        STEP = 11
        idxs = range(0,66,STEP)
        subset = [f[idxs] for f in features[:4]]
        allt = targets[0][idxs]
        imglen = 64*64*3
        img = allt[:,:imglen]
        img = np.reshape(img, (6,64,64,3))
        data, arms, grippers = self.predictor.predict(subset)
        for j in range(6):
            jj = j * STEP
            for k in range(min(4,self.num_hypotheses)):
                ax = axes[1+k][j]
                ax.set_axis_off()
                ax.imshow(np.squeeze(data[j][k]))
                ax.axis('off')
            ax = axes[0][j]
            ax.set_axis_off()
            ax.imshow(np.squeeze(features[0][jj]))
            ax.axis('off')
            ax = axes[-1][j]
            ax.set_axis_off()
            ax.imshow(np.squeeze(img[j]))
            ax.axis('off')

        plt.ion()
        plt.show(block=False)
        plt.pause(0.01)

    def _makeModel(self, features, arm, gripper, *args, **kwargs):
        self.predictor, self.train_predictor, self.actor = \
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
        q_target = goal_arm
        g_target = goal_gripper
        o_target = label

        print("sanity check:")
        print("-------------")
        print("images:", I.shape, I_target.shape)
        print("joints:", q.shape)
        print("options:", oin.shape, o_target.shape)

        if self.predictor is None:
            self._makeModel(I, q, g, qa, ga, oin)

        # ==============================
        image_shape = I.shape[1:]
        image_size = 1
        for dim in image_shape:
            image_size *= dim
        image_size = int(image_size)
        arm_size = q.shape[-1]
        gripper_size = g.shape[-1]

        train_size = image_size + arm_size + gripper_size
        assert gripper_size == 1
        assert train_size == 12295
        assert I.shape[0] == I_target.shape[0]

        length = I.shape[0]
        Itrain = np.reshape(I_target,(length, image_size))
        train_target = np.concatenate([Itrain,q_target,g_target],axis=-1)

        self.train_predictor.compile(
                loss=[
                    "mse","mse","mse",
                    MhpLossWithShape(
                        num_hypotheses=self.num_hypotheses,
                        outputs=[image_size, arm_size, gripper_size],
                        weights=[0.7,0.2,0.1],), 
                    #MhpLoss(
                    #    num_hypotheses=self.num_hypotheses,
                    #    num_outputs=train_size),
                    "mse","mse"],
                loss_weights=[0.1,0.1,0.1,0.5,0.1,0.1],
                optimizer=self.getOptimizer())
        self.predictor.compile(loss="mse", optimizer=self.getOptimizer())

        # ===============================================
        # Fit the main models
        self._fitPredictor(
                [I, q, g, oin, I_target, q_target, g_target,],
                #[I, q, g, oin, I_target, q_target, g_target, label],
                #[I, q, g, I_target, q_target, g_target],
                [I, q, g, np.expand_dims(train_target,axis=1), qa, ga],)

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

    def save(self):
        '''
        Save to a filename determined by the "self.name" field.
        '''
        if self.predictor is not None:
            print("----------------------------")
            print("Saving to " + self.name + "_{predictor, actor}")
            self.predictor.save_weights(self.name + "_predictor.h5f")
            if self.actor is not None:
                self.actor.save_weights(self.name + "_actor.h5f")
        else:
            raise RuntimeError('save() failed: model not found.')

    def _loadWeights(self, *args, **kwargs):
        '''
        Load model weights. This is the default load weights function; you may
        need to overload this for specific models.
        '''
        if self.predictor is not None:
            print("----------------------------")
            print("using " + self.name + " to load")
            try:
                self.actor.load_weights(self.name + "_actor.h5f")
            except Exception as e:
                print(e)
            self.predictor.load_weights(self.name + "_predictor.h5f")
        else:
            raise RuntimeError('_loadWeights() failed: model not found.')

    def predict(self, world):
        '''
        Evaluation for a feature-predictor model. This has two steps:
          - predict a set of features associated with the current world state
          - predict the expected reward based on each of those features
          - choose the best one to execute
        '''
        features = world.initial_features #getHistoryMatrix()
        if isinstance(features, list):
            assert len(features) == len(self.supervisor.inputs) - 1
        else:
            features = [features]
        features = [f.reshape((1,)+f.shape) for f in features]
        res = self.predictor.predict(features +
                [self._makeOption1h(self.prev_option)])
        print("# results = ", len(res))
        idx = np.random.randint(self.num_hypotheses)

        # Evaluate this policy to get the next action out
        return policy.predict(features)
