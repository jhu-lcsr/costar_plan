
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

    def _numLabels(self):
        '''
        Use the taskdef to get total number of labels
        '''
        return self.taskdef.numActions()

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

        x = Dense(self.combined_dense_size)(hidden)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)

        arm_out = Dense(arm_size)(x)
        gripper_out = Dense(gripper_size)(x)

        policy = Model(self.predictor.inputs, [arm_out, gripper_out])
        #model = Model(img_ins, [arm_out])
        optimizer = self.getOptimizer()
        policy.compile(loss="mse", optimizer=optimizer)

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
                kernel_size=[3,3],
                dense=False,
                tile=True,
                option=self._numLabels(),
                )
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

        # Predict the next joint states and gripper position. We add these back
        # in from the inputs once again, in order to make sure they don't get
        # lost in all the convolution layers above...
        x = Concatenate()([enc, ins[1], ins[2]])
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        arm_out = Dense(arm_size)(x)
        gripper_out = Dense(gripper_size)(x)

        # Predict the next option
        x = Dense(self.combined_dense_size)(enc)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        label_out = Dense(self._numLabels(), activation="sigmoid")(x)

        supervisor = Model(ins, [label_out])
        supervisor.compile(
                loss=["binary_crossentropy"],
                optimizer=self.getOptimizer())

        decoder = Model([rep], dec)
        features_out = [decoder(enc), arm_out, gripper_out,]
        predictor = Model(ins, features_out)
        predictor.compile(
                loss=["mse","mse","mse"],
                optimizer=self.getOptimizer())

        return x, supervisor, predictor

    def _fitPredictor(self, features, targets):
        if self.show_iter > 0:
            fig, axes = plt.subplots(4, 5,)

        self.predictor.trainable = True

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
                data = self.predictor.predict(features[0:5])
                for j in xrange(5):
                    jj = j * 5
                    ax = axes[1][j]
                    ax.imshow(np.squeeze(data[0][jj]))
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
                goal_features, goal_arm, goal_gripper], _ = \
            SplitIntoChunks(
                datasets=[features, arm, gripper, arm_cmd, gripper_cmd,
                    action_labels, goal_features, goal_arm, goal_gripper,],
                reward=None, reward_threshold=0.,
                labels=example,
                chunk_length=self.num_frames+2,
                step_size=self.partition_step_size,
                front_padding=True,
                rear_padding=False,
                start_off=1,
                end_off=1)

        # create inputs
        I = np.squeeze(features[:,1:-1])
        q = np.squeeze(arm[:,1:-1])
        g = np.squeeze(gripper[:,1:-1])
        qa = np.squeeze(arm_cmd[:,1:-1])
        ga = np.squeeze(gripper_cmd[:,1:-1])
        oin = np.squeeze(action_labels[:,:self.num_frames])
        o_target = np.squeeze(action_labels[:,1:-1])
        #I_target = np.squeeze(features[:,2:])
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

        #self._makeModel(features, arm, gripper, arm_cmd,
        #        gripper_cmd, action_labels, *args, **kwargs)
        self._makeModel(I, q, g, qa, ga, oin, *args, **kwargs)

        # Fit the main models
        self._fitPredictor([I, q, g, oin], [I_target, q_target, g_target])
        self._fitSupervisor([I, q, g, oin], o_target)

        action_target = [qa, ga]
        self._fitPolicies([features, arm, gripper], action_labels, action_target)
        self._fitBaseline([I, q, g, oin], action_target)


    def save(self):
        '''
        Store the model to disk here.
        '''
        pass
