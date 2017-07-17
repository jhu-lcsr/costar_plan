import keras.backend as K
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, Lambda
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from tensorflow import TensorShape

from abstract import AbstractAgentBasedModel
from robot_multi_models import *
from split import *
from trajectory import AddSamplerLayer, TrajectorySamplerLoss

class RobotMultiTrajectorySampler(AbstractAgentBasedModel):
    '''
    This creates an architecture that will generate a large number of possible
    trajectories that we could execute. It attempts to minimize the
    sample-based KL divergence from the target data when doing so.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        '''
        Read in taskdef for this model (or set of models). Use it to create the
        regression neural net that we can fit to compute our next action.

        Remember, here the "labels" are computed from the task model. We can
        use images and joint states together to compute next image or next
        joint state.
        '''

        super(RobotMultiTrajectorySampler, self).__init__(*args, **kwargs)

        self.taskdef = taskdef
        self.model = None
        
        self.dropout_rate = 0.5
        
        self.img_dense_size = 512
        self.img_col_dim = 256
        self.img_num_filters = 32
        self.robot_col_dense_size = 128
        self.robot_col_dim = 64
        self.combined_dense_size = 64
        self.decoder_filters = 16

        self.num_samples = 16
        self.trajectory_length = 10

    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            example, *args, **kwargs):
        '''
        Training data -- first, break into chunks of size "trajectory_length".
        In this case we actually don't care about the action labels, which we
        will still need to extract from the task model.
        
        Instead we are just going to try to fit a distribution over
        trajectories. Right now trajectory execution is not particularly noisy,
        so this should not be super hard.

        Parameters:
        -----------
        features: image features available at the beginning and end of the
        trajectory.
        arm: joint positions for the robot arm.
        gripper: gripper state.
        arm_cmd: goal positions sent out by the expert controller.
        gripper_cmd: gripper command sent out by the expert controller.
        label: string action description.
        example: iteration number; sampled environment.

        We ignore inputs including the reward (for now!)
        '''

        [features, arm, gripper, arm_cmd, gripper_cmd] = \
                SplitIntoChunks([features, arm, gripper, arm_cmd, gripper_cmd],
                example, self.trajectory_length, step_size=2)

        # Get images for input and output from the network.
        img_in = FirstInChunk(features)
        arm_in = FirstInChunk(arm)
        gripper_in = FirstInChunk(gripper)
        img_out = LastInChunk(features)

        img_shape = features.shape[2:]
        arm_size = arm.shape[-1]
        if len(gripper_in.shape) > 1:
            gripper_size = gripper_in.shape[-1]
        else:
            gripper_size = 1

        print "-------------------------------"
        print "KEY VARIABLES:"
        print "# arm features =", arm_size
        print "# gripper features =", gripper_size
        print "img data size =", features.shape
        print "img in size =", img_in.shape
        print "arm in size =", arm_in.shape
        print "gripper in size =", gripper_in.shape
        print "img out size =", img_out.shape
        print "-------------------------------"

        img_ins, img_out = GetCameraColumn(
                img_shape,
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                self.img_dense_size,)
        robot_ins, robot_out = GetArmGripperColumns(
                arm_size, 
                gripper_size,
                self.robot_col_dim,
                self.dropout_rate,
                self.robot_col_dense_size,)

        # Noise for sampling
        noise_in = Input((self.noise_dim,))
    
        x = Concatenate()([img_out, robot_out, noise_in])
        x = AddSamplerLayer(x,
                int(self.num_samples),
                int(self.trajectory_length/4),
                self.decoder_filters)
        x = UpSampling2D(size=(1,2))(x)
        x = Conv2D(self.decoder_filters, 3, 3, border_mode='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(1,2))(x)
        x = Conv2D(arm_size, 3, 3, border_mode='same')(x)

        # s0 is the initial state. it needs to be repeated num_samples *
        # traj_length times.
        s0 = Reshape((1,1,6))(robot_ins[0])
        s0 = K.tile(s0,
                TensorShape([1,self.num_samples,self.trajectory_length,1]))

        # Integrate along the trajectories
        x = Lambda(lambda x: K.cumsum(x, axis=2) + s0)(x)


        arm_loss = TrajectorySamplerLoss(self.num_samples,
                    self.trajectory_length, arm_size)

        self.model = Model(img_ins + robot_ins + [noise_in], x)
        self.model.summary()
        self.model.compile(optimizer=self.getOptimizer(), 
                loss=arm_loss)

        for i in xrange(self.iter):
            idx = np.random.randint(0, img_in.shape[0], size=self.batch_size)
            xi = img_in[idx]
            xa = arm_in[idx]
            xg = gripper_in[idx]

            # create targets
            y_shape = (self.batch_size,1)+arm_cmd.shape[1:]
            ya = np.reshape(arm_cmd[idx],y_shape)
            
            # duplicate
            ya = ya[:,np.zeros((self.num_samples,),dtype=int)]

            noise = np.random.random((self.batch_size, self.noise_dim))
            loss = self.model.train_on_batch([xi, xa, xg, noise], ya)
            print "Iter %d: loss = %f"%(i,loss)

    def save(self):
        if self.model is not None:
            self.model.save_weights(self.name + ".h5f")

    def load(self):
        self.model.load_weights(self.name + ".h5f")

    def plot(self, env):
        fig = plt.figure()
        for i in xrange(9):
            env.reset()
            actor = env.world.actors[0]
            robot = actor.robot
            q = np.array([actor.state.arm])
            g = np.array([actor.state.gripper])
            I = np.array([env.world.cameras[0].capture().rgb])
            z = noise = np.random.random((1, self.noise_dim))
            print "Debug shapes:"
            print q.shape, g.shape, I.shape, z.shape
            trajs = self.model.predict([I, q, g, z])[0]
            print "output trajectories:"
            print trajs.shape
            print trajs
            trajs3d = []
            for traj in trajs:
                fwd_traj = []
                for pt in traj:
                    fwd_traj.append(list(robot.fwd(pt).p))
                trajs3d.append(np.array(fwd_traj))

            ax = fig.add_subplot(3,3,i+1, projection='3d')
            for traj in trajs3d:
                plt.plot(traj[:,0], traj[:,1], traj[:,2])
        plt.show()
