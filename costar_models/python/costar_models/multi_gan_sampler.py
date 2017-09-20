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

class RobotMultiGanSampler(RobotMultiHierarchical):

    '''
    This class is set up as a SUPERVISED learning problem -- for more
    interactive training we will need to add data from an appropriate agent.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(RobotMultiGanSampler, self).__init__(taskdef, *args, **kwargs)

        self.num_frames = 1

        self.dropout_rate = 0.5
        self.img_col_dim = 512
        self.img_num_filters = 64
        self.tform_filters = 32
        self.combined_dense_size = 128
        self.num_hypotheses = 8
        self.num_transforms = 2
        self.validation_split = 0.1
        self.num_options = 48
        self.extra_layers = 0

        self.predictor = None
        self.train_predictor = None
        self.actor = None

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
                dropout=True,
                pre_tiling_layers=self.extra_layers,
                post_tiling_layers=3,
                kernel_size=[5,5],
                dense=False,
                batchnorm=True,
                tile=True,
                option=self.num_options,
                flatten=False,
                output_filters=self.tform_filters,
                )
        decoder = GetImageArmGripperDecoder(self.img_col_dim,
                        img_shape,
                        dropout_rate=self.dropout_rate,
                        dense_size=self.combined_dense_size,
                        kernel_size=[5,5],
                        filters=self.img_num_filters,
                        stride2_layers=3,
                        stride1_layers=self.extra_layers,
                        tform_filters=self.tform_filters,
                        num_options=self.num_options,
                        arm_size=arm_size,
                        gripper_size=gripper_size,
                        dropout=False,
                        leaky=True,
                        dense=False,
                        resnet_blocks=self.residual,
                        batchnorm=True,)
        dins, discriminator = GetEncoder(img_shape,
                arm_size,
                gripper_size,
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                discriminator=True,
                leaky=False,
                dropout=True,
                pre_tiling_layers=self.extra_layers,
                post_tiling_layers=3,
                kernel_size=[5,5],
                dense=False,
                batchnorm=True,
                tile=True,
                option=self.num_options,
                flatten=False,
                output_filters=self.tform_filters,
                )
        decoder.summary()

        # =====================================================================
        # Get the transform
        transform = GetTranform(
                rep_size=(8,8),
                filters=self.tform_filters,
                kernel_size=[5,5],
                idx=i,
                batchnorm=True,
                leaky=True,
                num_blocks=self.num_transforms,
                relu=True,
                resnet_blocks=self.residual,)
        if i == 0:
            transform.summary()
        x = transform([enc])
            
        # This maps from our latent world state back into observable images.
        #decoder = Model(rep, dec)
        img_x, arm_x, gripper_x, label_x = decoder([x])

        # =====================================================================
        # Create models to train
        predictor = Model(ins, [image_out, arm_out, gripper_out, label_out])
        discriminator = Model(dins, discriminator)
        return predictor, discriminator

    def _fitPredictor(self, features, targets,):
        if self.show_iter > 0:
            fig, axes = plt.subplots(6, 6,)
            plt.tight_layout()

        image_shape = features[0].shape[1:]
        image_size = 1.
        for dim in image_shape:
            image_size *= dim

        for i in range(features[0].shape[0]):
            img1 = targets[0][i,:int(image_size)].reshape((64,64,3))
            img2 = features[4][i]
            if not np.all(img1 == img2):
                print(i,"failed")
                plt.subplot(1,2,1); plt.imshow(img1);
                plt.subplot(1,2,2); plt.imshow(img2);
                plt.show()

        if self.show_iter == 0 or self.show_iter == None:
            modelCheckpointCb = ModelCheckpoint(
                filepath=self.name+"_train_predictor_weights.h5f",
                verbose=1,
                save_best_only=False # does not work without validation wts
            )
            imageCb = PredictorShowImage(
                self.predictor,
                features=features[:4],
                targets=targets,
                num_hypotheses=self.num_hypotheses,
                verbose=True,
                min_idx=0,
                max_idx=5,
                step=1,)
            self.train_predictor.fit(features,
                    [np.expand_dims(f,1) for f in targets],
                    callbacks=[modelCheckpointCb, imageCb],
                    validation_split=self.validation_split,
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
                yimg = y[0][:,0,:int(image_size)]
                yimg = yimg.reshape((self.batch_size,64,64,3))
                for j in range(self.batch_size):
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
        data, arms, grippers, labels = self.predictor.predict(subset)
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
        '''
        Little helper function wraps makePredictor to consturct all the models.

        Parameters:
        -----------
        features, arm, gripper: variables of the appropriate sizes
        '''
        self.predictor, self.discriminator = \
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

        train_size = image_size + arm_size + gripper_size + self.num_options
        assert gripper_size == 1
        assert train_size == 12295 + self.num_options
        assert I.shape[0] == I_target.shape[0]

        o_target = np.squeeze(self.toOneHot2D(o_target, self.num_options))
        length = I.shape[0]
        Itrain = np.reshape(I_target,(length, image_size))
        train_target = np.concatenate([Itrain,q_target,g_target,o_target],axis=-1)

        self.train_predictor.compile(
                loss=[
                    MhpLossWithShape(
                        num_hypotheses=self.num_hypotheses,
                        outputs=[image_size, arm_size, gripper_size, self.num_options],
                        weights=[0.7,0.2,0.05,0.05],
                        loss=["mse","mse","mse","categorical_crossentropy"]), 
                    ],#"mse","mse"],
                #loss_weights=[0.8,0.1,0.1],
                optimizer=self.getOptimizer())
        self.predictor.compile(loss="mse", optimizer=self.getOptimizer())

        # ===============================================
        # Fit the main models
        self._fitPredictor(
                [I, q, g, oin, I_target, q_target, g_target,],
                [train_target,]), #qa, ga],)

    def _getAllData(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            prev_label, goal_features, goal_arm, goal_gripper, *args, **kwargs):
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

        # ==============================
        image_shape = I.shape[1:]
        image_size = 1
        for dim in image_shape:
            image_size *= dim
        image_size = int(image_size)
        arm_size = q.shape[-1]
        gripper_size = g.shape[-1]

        train_size = image_size + arm_size + gripper_size + self.num_options
        assert gripper_size == 1
        #assert train_size == 12295 + self.num_options
        # NOTE: arm size is one bigger when we have quaternions
        assert train_size == 12296 + self.num_options
        assert I.shape[0] == I_target.shape[0]

        o_target = np.squeeze(self.toOneHot2D(o_target, self.num_options))
        length = I.shape[0]
        Itrain = np.reshape(I_target,(length, image_size))
        train_target = np.concatenate([Itrain,q_target,g_target,o_target],axis=-1)

        return [I, q, g, oin, I_target, q_target, g_target,], [
                np.expand_dims(train_target, axis=1),
                np.expand_dims(qa, axis=1),
                np.expand_dims(ga, axis=1)]

    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        return features[:4], [targets[0]]

    def trainFromGenerators(self, train_generator, test_generator, data=None):
        '''
        Train tool from generator

        Parameters:
        -----------
        train_generator: produces training examples
        test_generator: produces test examples
        data: some extra data used for debugging (should be validation data)
        '''
        if data is not None:
            features, targets = self._getAllData(**data)
        else:
            raise RuntimeError('predictor model sets sizes based on'
                               'sample data; must be provided')
        # ===================================================================
        # Use sample data to compile the model and set everything else up.
        # Check to make sure data makes sense before running the model.

        [I, q, g, oin, I_target, q_target, g_target,] = features
        [I_target2, qa, ga,] = targets

        if self.predictor is None:
            self._makeModel(I, q, g, qa, ga, oin)

        # Compute helpful variables
        image_shape = I.shape[1:]
        image_size = 1
        for dim in image_shape:
            image_size *= dim
        image_size = int(image_size)
        arm_size = q.shape[-1]
        gripper_size = g.shape[-1]

        train_size = image_size + arm_size + gripper_size + self.num_options
        assert gripper_size == 1
        # NOTE: arm size is one bigger when we have quaternions
        #assert train_size == 12295 + self.num_options
        assert train_size == 12296 + self.num_options
        self.train_predictor.compile(
                loss=[
                    MhpLossWithShape(
                        num_hypotheses=self.num_hypotheses,
                        outputs=[image_size, arm_size, gripper_size, self.num_options],
                        weights=[0.3,0.3,0.1,0.3],
                        loss=["mse","mse","mse","categorical_crossentropy"]),
                    ],
                    #"mse","mse"],
                #loss_weights=[0.8,0.1,0.1],
                optimizer=self.getOptimizer())
        self.predictor.compile(loss="mse", optimizer=self.getOptimizer())

        # ===================================================================
        # Create the callbacks and actually run the training loop.
        modelCheckpointCb = ModelCheckpoint(
            filepath=self.name+"_predictor_weights.h5f",
            verbose=1,
            save_best_only=True # does not work without validation wts
        )
        imageCb = PredictorShowImage(
            self.predictor,
            features=features[:4],
            targets=targets,
            num_hypotheses=self.num_hypotheses,
            verbose=True,
            min_idx=0,
            max_idx=5,
            step=1,)
        self.train_predictor.fit_generator(
                train_generator,
                self.steps_per_epoch,
                epochs=self.epochs,
                validation_steps=self.validation_steps,
                validation_data=test_generator,
                callbacks=[modelCheckpointCb, imageCb])

    def save(self):
        '''
        Save to a filename determined by the "self.name" field.
        '''
        if self.predictor is not None:
            print("----------------------------")
            print("Saving to " + self.name + "_{predictor, actor}")
            self.train_predictor.save_weights(self.name + "_train_predictor.h5f")
            if self.actor is not None:
                self.predictor.save_weights(self.name + "_predictor.h5f")
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
                #self.predictor.load_weights(self.name + "_predictor.h5f")
            except Exception as e:
                print(e)
            self.train_predictor.load_weights(self.name + "_train_predictor.h5f")
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

