
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

        self.num_frames = 1

        self.dropout_rate = 0.5
        self.img_dense_size = 1024
        self.img_col_dim = 512
        self.img_num_filters = 128
        self.combined_dense_size = 128
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

        policy = Model(self.supervisor.inputs, [arm_out, gripper_out])

        return policy

    def _makeSupervisor(self, features):
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
                tile=True,
                option=None,#self._numLabels(),
                flatten=False,
                )

        # Tile on the option -- this is where our transition model comes in
        tile_width = img_shape[0]/(2**3)
        tile_height = img_shape[1]/(2**3)
        tile_shape = (1, tile_width, tile_height, 1)
        option_in = Input((self._numLabels(),))
        option = Reshape([1,1,self._numLabels()])(option_in)
        option = Lambda(lambda x: K.tile(x, tile_shape))(option)
        enc_with_option = Concatenate(axis=-1)([enc,option])
        enc_with_option = Conv2D(self.img_num_filters,
                kernel_size=[5,5], 
                strides=(1, 1),
                padding='same')(enc_with_option)
        ins.append(option_in)
        
        rep, dec = GetDecoder(self.img_col_dim,
                            img_shape,
                            arm_size,
                            gripper_size,
                            dropout_rate=self.dropout_rate,
                            kernel_size=[5,5],
                            filters=self.img_num_filters,
                            stride2_layers=3,
                            stride1_layers=0,
                            dropout=False,
                            leaky=True,
                            dense=False,
                            batchnorm=True,)
        rep2, dec2 = GetDecoder(self.img_col_dim,
                            img_shape,
                            arm_size,
                            gripper_size,
                            dropout_rate=self.dropout_rate,
                            kernel_size=[5,5],
                            filters=self.img_num_filters,
                            stride2_layers=3,
                            stride1_layers=0,
                            dropout=False,
                            leaky=True,
                            dense=False,
                            batchnorm=True,)

        # Predict the next joint states and gripper position. We add these back
        # in from the inputs once again, in order to make sure they don't get
        # lost in all the convolution layers above...
        x = Conv2D(self.img_num_filters/2,
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(enc_with_option)
        x = Flatten()(x)
        x = Concatenate()([x, ins[1], ins[2]])
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        arm_out = Dense(arm_size,name="action_arm_goal")(x)
        gripper_out = Dense(gripper_size,name="action_gripper_goal")(x)

        # =====================================================================
        # SUPERVISOR
        # Predict the next option -- does not depend on option
        x = Conv2D(self.img_num_filters/4,
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(enc)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        label_out = Dense(self._numLabels(), activation="sigmoid")(x)

        supervisor = Model(ins[:3], [label_out])

        enc_with_option_flat = Flatten()(enc_with_option)
        decoder = Model([rep], dec, name="action_image_goal_decoder")
        next_frame_decoder = Model(
                [rep2],
                dec2,
                name="action_next_image_decoder")
        features_out = [
                decoder(enc_with_option_flat),
                arm_out,
                gripper_out,
                next_frame_decoder(enc_with_option_flat)]
        predictor = Model(ins, features_out)

        return enc, supervisor, predictor

    def _fitPredictor(self, features, targets):
        if self.show_iter > 0:
            fig, axes = plt.subplots(5, 5,)

        self._unfixWeights()
        self.predictor.compile(
                loss="mse",
                optimizer=self.getOptimizer())
        self.predictor.summary()

        for i in xrange(self.iter):
            idx = np.random.randint(0, features[0].shape[0], size=self.batch_size)
            x = []
            y = []
            for f in features:
                x.append(f[idx])
            for f in targets:
                y.append(f[idx])

            losses = self.predictor.train_on_batch(x, y)

            print "Iter %d: loss ="%(i),losses
            if self.show_iter > 0 and (i+1) % self.show_iter == 0:
                self.plotInfo(features, targets, axes)

        self._fixWeights()

    def plotInfo(self, features, targets, axes):
            data = self.predictor.predict(features[0:5])
            for j in xrange(5):
                jj = j * 5
                ax = axes[1][j]
                ax.imshow(np.squeeze(data[0][jj]))
                ax.axis('off')
                ax = axes[4][j]
                ax.imshow(np.squeeze(data[3][jj]))
                ax.axis('off')
                ax = axes[0][j]
                ax.imshow(np.squeeze(features[0][jj]))
                ax.axis('off')
                ax = axes[2][j]
                ax.imshow(np.squeeze(targets[0][jj]))
                ax.axis('off')
                
                q0 = features[1][jj]
                q = data[1][j]
                q1 = targets[1][jj]
                ax = axes[3][j]
                ax.bar(np.arange(6),q0,1./3.,color='b')
                ax.bar(np.arange(6)+1./3.,q,1./3.,color='r')
                ax.bar(np.arange(6)+2./3.,q1,1./3.,color='g')

            plt.ion()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)


    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            example, reward, *args, **kwargs):
        '''
        Pre-process training data.

        Then, create the model. Train based on labeled data. Remove
        unsuccessful examples.
        '''
        action_labels_num = np.array([self.taskdef.index(l) for l in label])
        action_labels = np.squeeze(self.toOneHot2D(action_labels_num,
            len(self.taskdef.indices)))


        [goal_features, goal_arm, goal_gripper] = NextAction( \
                [features, arm, gripper],
                action_labels_num,
                example)


        [features, arm, gripper, arm_cmd, gripper_cmd, action_labels,
                goal_features, goal_arm, goal_gripper] = \
            SplitIntoChunks(
                datasets=[features, arm, gripper, arm_cmd, gripper_cmd,
                    action_labels, goal_features, goal_arm, goal_gripper,],
                reward=None, reward_threshold=0.,
                labels=example,
                chunk_length=self.num_frames+2,
                front_padding=True,
                rear_padding=False,)

        # create inputs
        I = np.squeeze(features[:,1:-1])
        q = np.squeeze(arm[:,1:-1])
        g = np.squeeze(gripper[:,1:-1])
        qa = np.squeeze(arm_cmd[:,1:-1])
        ga = np.squeeze(gripper_cmd[:,1:-1])
        #oin = np.squeeze(action_labels[:,:self.num_frames])
        oin = np.squeeze(action_labels[:,1:-1])
        o_target = np.squeeze(action_labels[:,1:-1])
        Inext_target = np.squeeze(features[:,2:])
        #q_target = np.squeeze(arm[:,2:])
        #g_target = np.squeeze(gripper[:,2:])
        # I_target = np.
        I_target = np.squeeze(goal_features[:,1:-1])
        q_target = np.squeeze(goal_arm[:,1:-1])
        g_target = np.squeeze(goal_gripper[:,1:-1])

        print "sanity check:",
        print "images:", I.shape, I_target.shape
        print "joints:", q.shape,
        print "options:", oin.shape, o_target.shape

        if False:
            # show the before and after frames
            for i in xrange(10):
                for j in xrange(self.num_frames+2):
                    plt.subplot(10,
                            self.num_frames+3,((self.num_frames+3)*i) + j + 1)
                    plt.imshow(features[i*5,j])
                    plt.axis('off')
                plt.subplot(10,self.num_frames+3,((self.num_frames+3)*i)+self.num_frames+3)
                plt.axis('off')
                plt.imshow(goal_features[i*5,1])
            plt.show()

        if self.supervisor is None:
            self._makeModel(I, q, g, qa, ga, oin, *args, **kwargs)

        # Fit the main models
        self._fitPredictor(
                [I, q, g, oin],
                [I_target, q_target, g_target, Inext_target])

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
        self._fitSupervisor([I, q, g], o_target)
        # ===============================================

        action_target = [qa, ga]
        self._fitPolicies([I, q, g], action_labels, action_target)
        self._fitBaseline([I, q, g], action_target)

