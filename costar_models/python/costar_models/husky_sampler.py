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

from .split import *
from .mhp_loss import *

class HuskyRobotMultiPredictionSampler(RobotMultiHierarchical):

    '''
    This class is set up as a SUPERVISED learning problem -- for more
    interactive training we will need to add data from an appropriate agent.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(HuskyRobotMultiPredictionSampler, self).__init__(taskdef, *args, **kwargs)

        self.num_frames = 1

        self.dropout_rate = 0.5
        self.img_col_dim = 512
        self.img_num_filters = 64
        self.tform_filters = 64
        self.combined_dense_size = 128
        self.num_hypotheses = 8
        self.num_transforms = 3
        self.num_options = 4

        self.predictor = None
        self.train_predictor = None
        self.actor = None
        
    def _TilePose(self, x, pose_in, tile_width, tile_height,
        option=None, option_in=None,
        time_distributed=None):
        pose_size = int(pose_in.shape[-1])
        
    
        # handle error: options and grippers
        if option is None and option_in is not None \
            or option is not None and option_in is None:
                raise RuntimeError('must provide both #opts and input')
    
        # generate options and tile things together
        if option is None:
            robot = pose_in
            reshape_size = pose_size
        else:
            robot = Concatenate(axis=-1)([pose_in, option_in])
            reshape_size = pose_size+option
    
        # time distributed or not
        if time_distributed is not None and time_distributed > 0:
            tile_shape = (1, 1, tile_width, tile_height, 1)
            robot = Reshape([time_distributed, 1, 1, reshape_size])(robot)
        else:
            tile_shape = (1, tile_width, tile_height, 1)
            robot = Reshape([1, 1, reshape_size])(robot)
    
        # finally perform the actual tiling
        robot = Lambda(lambda x: K.tile(x, tile_shape))(robot)
        x = Concatenate(axis=-1)([x,robot])
    
        return x
    
    def _GetImageEncoder(self, img_shape, dim, dropout_rate,
            filters, dropout=True, leaky=True,
            dense=True, flatten=True,
            layers=2,
            kernel_size=[3,3],
            time_distributed=0,):
    
        if time_distributed <= 0:
            ApplyTD = lambda x: x
            height4 = img_shape[0]/4
            width4 = img_shape[1]/4
            height2 = img_shape[0]/2
            width2 = img_shape[1]/2
            height = img_shape[0]
            width = img_shape[1]
            channels = img_shape[2]
        else:
            ApplyTD = lambda x: TimeDistributed(x)
            height4 = img_shape[1]/4
            width4 = img_shape[2]/4
            height2 = img_shape[1]/2
            width2 = img_shape[2]/2
            height = img_shape[1]
            width = img_shape[2]
            channels = img_shape[3]
    
        samples = Input(shape=img_shape)
    
        '''
        Convolutions for an image, terminating in a dense layer of size dim.
        '''
    
        if leaky:
            relu = lambda: LeakyReLU(alpha=0.2)
        else:
            relu = lambda: Activation('relu')
    
        x = samples
    
        x = ApplyTD(Conv2D(filters,
                    kernel_size=kernel_size, 
                    strides=(1, 1),
                    padding='same'))(x)
        x = ApplyTD(relu())(x)
        if dropout:
            x = ApplyTD(Dropout(dropout_rate))(x)
    
        for i in range(layers):
    
            x = ApplyTD(Conv2D(filters,
                       kernel_size=kernel_size, 
                       strides=(2, 2),
                       padding='same'))(x)
            x = ApplyTD(relu())(x)
            if dropout:
                x = ApplyTD(Dropout(dropout_rate))(x)
    
        if flatten or dense:
            x = ApplyTD(Flatten())(x)
        if dense:
            x = ApplyTD(Dense(dim))(x)
            x = ApplyTD(relu())(x)
    
        return [samples], x
        
    def _GetEncoder(self, img_shape, pose_size, dim, dropout_rate,
        filters, discriminator=False, tile=False, dropout=True, leaky=True,
        dense=True, option=None, flatten=True, batchnorm=False,
        pre_tiling_layers=0,
        post_tiling_layers=2,
        kernel_size=[3,3], output_filters=None,
        time_distributed=0,):


        if output_filters is None:
            output_filters = filters
    
        if time_distributed <= 0:
            ApplyTD = lambda x: x
            pose_in = Input((pose_size,))
          
            if option is not None:
                option_in = Input((1,))
                option_x = OneHot(size=option)(option_in)
                option_x = Reshape((option,))(option_x)
            else:
                option_in, option_x = None, None
            print ("img_shape", img_shape)
            height4 = img_shape[0]/4
            width4 = img_shape[1]/4
            height2 = img_shape[0]/2
            width2 = img_shape[1]/2
            height = img_shape[0]
            width = img_shape[1]
            channels = img_shape[2]
        else:
            ApplyTD = lambda x: TimeDistributed(x)
            pose_in = Input((time_distributed, pose_size,))
           
            if option is not None:
                option_in = Input((time_distributed,1,))
                option_x = TimeDistributed(OneHot(size=option),name="label_to_one_hot")(option_in)
                option_x = Reshape((time_distributed,option,))(option_x)
            else:
                option_in, option_x = None, None
            height4 = img_shape[1]/4
            width4 = img_shape[2]/4
            height2 = img_shape[1]/2
            width2 = img_shape[2]/2
            height = img_shape[1]
            width = img_shape[2]
            channels = img_shape[3]
    
        samples = Input(shape=img_shape)
    
        '''
        Convolutions for an image, terminating in a dense layer of size dim.
        '''
    
        if leaky:
            relu = lambda: LeakyReLU(alpha=0.2)
        else:
            relu = lambda: Activation('relu')
    
        x = samples
    
        x = ApplyTD(Conv2D(filters,
                    kernel_size=kernel_size, 
                    strides=(1, 1),
                    padding='same'))(x)
        x = ApplyTD(relu())(x)
        if batchnorm:
            x = ApplyTD(BatchNormalization(momentum=0.9))(x)
        if dropout:
            x = ApplyTD(Dropout(dropout_rate))(x)
    
        for i in range(pre_tiling_layers):
    
            x = ApplyTD(Conv2D(filters,
                       kernel_size=kernel_size, 
                       strides=(2, 2),
                       padding='same'))(x)
            if batchnorm:
                x = ApplyTD(BatchNormalization(momentum=0.9))(x)
            x = ApplyTD(relu())(x)
            #x = MaxPooling2D(pool_size=(2,2))(x)
            if dropout:
                x = ApplyTD(Dropout(dropout_rate))(x)
    
        # ===============================================
        # ADD TILING
        if tile:
            tile_width = int(width/(pre_tiling_layers+1))
            tile_height = int(height/(pre_tiling_layers+1))
            if option is not None:
                ins = [samples, pose_in, option_in]
            else:
                ins = [samples, pose_in]
            x = self._TilePose(x, pose_in, tile_height, tile_width,
                    option, option_x, time_distributed)
        else:
            ins = [samples]
    
        for i in range(post_tiling_layers):
            if i == post_tiling_layers - 1:
                nfilters = output_filters
            else:
                nfilters = filters
            x = ApplyTD(Conv2D(nfilters,
                       kernel_size=kernel_size, 
                       strides=(2, 2),
                       padding='same'))(x)
            if batchnorm:
                x = ApplyTD(BatchNormalization(momentum=0.9))(x)
            x = relu()(x)
            #x = MaxPooling2D(pool_size=(2,2))(x)
            if dropout:
                x = Dropout(dropout_rate)(x)
    
        if flatten or dense or discriminator:
            x = ApplyTD(Flatten())(x)
        if dense:
            x = ApplyTD(Dense(dim))(x)
            x = ApplyTD(relu())(x)
    
        # Single output -- sigmoid activation function
        if discriminator:
            x = Dense(1,activation="sigmoid")(x)
    
        return ins, x
    
    def _AddOptionTiling(self, x, option_length, option_in, height, width):
        tile_shape = (1, width, height, 1)
        option = Reshape([1,1,option_length])(option_in)
        option = Lambda(lambda x: K.tile(x, tile_shape))(option)
        x = Concatenate(
                axis=-1,
                name="add_option_%dx%d"%(width,height),
            )([x, option])
        return x
    
    def _GetDecoder(self, dim, img_shape, pose_size,
            dropout_rate, filters, kernel_size=[3,3], dropout=True, leaky=True,
            batchnorm=True,dense=True, option=None, num_hypotheses=None,
            tform_filters=None,
            stride2_layers=2, stride1_layers=1):
    
        '''
        Initial decoder: just based on getting images out of the world state
        created via the encoder.
        '''
    
        height8 = img_shape[0]/8
        width8 = img_shape[1]/8
        height4 = img_shape[0]/4
        width4 = img_shape[1]/4
        height2 = img_shape[0]/2
        width2 = img_shape[1]/2
        nchannels = img_shape[2]
    
        if tform_filters is None:
            tform_filters = filters
    
        if leaky:
            relu = lambda: LeakyReLU(alpha=0.2)
        else:
            relu = lambda: Activation('relu')
    
        if option is not None:
            oin = Input((1,),name="input_next_option")
    
        if dense:
            z = Input((dim,),name="input_image")
            x = Dense(filters/2 * height4 * width4)(z)
            if batchnorm:
                x = BatchNormalization(momentum=0.9)(x)
            x = relu()(x)
            x = Reshape((width4,height4,tform_filters/2))(x)
        else:
            z = Input((width8*height8*tform_filters,),name="input_image")
            x = Reshape((width8,height8,tform_filters))(z)
        x = Dropout(dropout_rate)(x)
    
        height = height4
        width = width4
        for i in range(stride2_layers):
    
            x = Conv2DTranspose(filters,
                       kernel_size=kernel_size, 
                       strides=(2, 2),
                       padding='same')(x)
            if batchnorm:
                x = BatchNormalization(momentum=0.9)(x)
            x = relu()(x)
            #x = UpSampling2D(size=(2,2))(x)
            if dropout:
                x = Dropout(dropout_rate)(x)
    
            if option is not None:
                opt = OneHot(option)(oin)
                x = AddOptionTiling(x, option, opt, height, width)
    
            height *= 2
            width *= 2
    
        for i in range(stride1_layers):
            x = Conv2D(filters, # + num_labels
                       kernel_size=kernel_size, 
                       strides=(1, 1),
                       padding="same")(x)
            if batchnorm:
                x = BatchNormalization(momentum=0.9)(x)
            x = relu()(x)
            if dropout:
                x = Dropout(dropout_rate)(x)
            if option is not None:
                opt = OneHot(option)(oin)
                x = AddOptionTiling(x, option, opt, height, width)
    
        x = Conv2D(nchannels, (1, 1), padding='same')(x)
        x = Activation('sigmoid')(x)
    
        ins = [z]
        if option is not None:
            ins.append(oin)
    
        return ins, x


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
        action: input list of action outputs (pose commands for the
                robot tasks).
        hidden: "hidden" embedding of latent world state (input)
        '''
        images, pose = features
        pose_cmd = action
        img_shape = images.shape[1:]
        pose_size = pose.shape[-1]
        
        

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

        pose_out = Dense(pose_size)(x)

        policy = Model(self.supervisor.inputs[:3], [pose_out])

        return policy

    def _makePredictor(self, features):
        '''
        Create model to predict possible manipulation goals.
        '''
        (images, pose) = features
        img_shape = images.shape[1:]
        pose_size = pose.shape[-1]
        

        ins, enc = self._GetEncoder(img_shape,
                pose_size,
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                leaky=False,
                dropout=True,
                pre_tiling_layers=0,
                post_tiling_layers=3,
                kernel_size=[5,5],
                dense=False,
                batchnorm=True,
                tile=True,
                option=self.num_options,
                flatten=False,
                output_filters=self.tform_filters,
                )
        gins, genc = self._GetEncoder(img_shape,
                pose_size,
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                leaky=False,
                dropout=True,
                pre_tiling_layers=0,
                post_tiling_layers=3,
                kernel_size=[5,5],
                dense=False,
                batchnorm=True,
                tile=True,
                #option=self.num_options,
                flatten=False,
                output_filters=self.tform_filters,
                )


        image_outs = []
        #pose_outs = []
        #gripper_outs = []
        pose_outs = [] # xyz y        
        
        train_outs = []
        label_outs = []

        
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
        # Decode pose/gripper state.
        # Predict the next joint states and gripper position. We add these back
        # in from the inputs once again, in order to make sure they don't get
        # lost in all the convolution layers above...
        
        #assuming 8x8, need to update the strides/pooling to make sense        
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
        pose_out_x = Dense(pose_size,name="next_pose")(x)
       
        label_out_x = Dense(self.num_options,name="next_label",activation="softmax")(x)

        decoder = Model(rep, [dec, pose_out_x, label_out_x], name="decoder")

        # =====================================================================
        # Create many different image decoders

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
            img_x, pose_x, label_x = decoder(x)

            # Create the training outputs
            train_x = Concatenate(axis=-1,name="combine_train_%d"%i)([
                            Flatten(name="flatten_img_%d"%i)(img_x),
                            pose_x,
                            label_x])
            img_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="img_hypothesis_%d"%i)(img_x)
            pose_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="pose_hypothesis_%d"%i)(pose_x)
            label_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="label_hypothesis_%d"%i)(label_x)
            train_x = Lambda(
                    lambda x: K.expand_dims(x, 1),
                    name="flattened_hypothesis_%d"%i)(train_x)

            image_outs.append(img_x)
            pose_outs.append(pose_x)
            #gripper_outs.append(gripper_x)
            label_outs.append(label_x)
            train_outs.append(train_x)

        image_out = Concatenate(axis=1)(image_outs)
        pose_out = Concatenate(axis=1)(pose_outs)
        #gripper_out = Concatenate(axis=1)(gripper_outs)
        label_out = Concatenate(axis=1)(label_outs)
        train_out = Concatenate(axis=1,name="all_train_outs")(train_outs)

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
        pose_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="pose_action")(
                Dense(pose_size)(y))
        

        # =====================================================================
        # Create models to train

        #predictor = Model(ins, [decoder(enc), pose_out, gripper_out])
        predictor = Model(ins, [image_out, pose_out, label_out])
        actor = Model(ins + gins, [pose_cmd_out])
        train_predictor = Model(ins + gins, [train_out,
                                             pose_cmd_out,
                                             ])

        return predictor, train_predictor, actor

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
            img2 = features[3][i]
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
                features=features[:3],
                targets=targets,
                num_hypotheses=self.num_hypotheses,
                verbose=True,
                min_idx=0,
                max_idx=1500,
                step=20,)
            self.train_predictor.fit(features,
                    [np.expand_dims(f,1) for f in targets],
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
        #STEP = 20
        #idxs = range(0,120,STEP)
        STEP = 1
        idxs = range(0,700,STEP)
        subset = [f[idxs] for f in features[:3]]
        allt = targets[0][idxs]
        imglen = 64*64*3
        img = allt[:,:imglen]
        img = np.reshape(img, (6,64,64,3))
        data, poses, labels = self.predictor.predict(subset)
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

    def _makeModel(self, features, pose, *args, **kwargs):
        self.predictor, self.train_predictor, self.actor = \
            self._makePredictor(
                (features, pose))

    def train(self, features, pose, pose_cmd, label,
            prev_label, goal_features, goal_pose, *args, **kwargs):
        '''
        Pre-process training data.

        Then, create the model. Train based on labeled data. Remove
        unsuccessful examples.
        '''

        I = features
        q = pose
        qa = pose_cmd
        oin = prev_label
        I_target = goal_features
        q_target = goal_pose
        o_target = label

        print("sanity check:")
        print("-------------")
        print("images:", I.shape, I_target.shape)
        print("joints:", q.shape)
        print("options:", oin.shape, o_target.shape)

        if self.predictor is None:
            self._makeModel(I, q, qa, oin)

        # ==============================
        image_shape = I.shape[1:]
        image_size = 1
        for dim in image_shape:
            image_size *= dim
        image_size = int(image_size)
        pose_size = q.shape[-1]

        train_size = image_size + pose_size + self.num_options
        #assert train_size == 12295 + 64
        assert train_size == 64*64*3 + 6 + self.num_options
        assert I.shape[0] == I_target.shape[0]

        o_target = np.squeeze(self.toOneHot2D(o_target, self.num_options))
        length = I.shape[0]
        Itrain = np.reshape(I_target,(length, image_size))
        train_target = np.concatenate([Itrain,q_target,o_target],axis=-1)

        self.train_predictor.compile(
                loss=[
                    MhpLossWithShape(
                        num_hypotheses=self.num_hypotheses,
                        outputs=[image_size, pose_size, self.num_options],
                        #weights=[0.6,0.3,0.1],
                        weights=[1.,0.,0.],
                        loss=["mse","mse","categorical_crossentropy"]), 
                    "mse"],
                loss_weights=[0.8,0.1],
                optimizer=self.getOptimizer())
        self.predictor.compile(loss="mse", optimizer=self.getOptimizer())

        # ===============================================
        # Fit the main models
        self._fitPredictor(
                [I, q, oin, I_target, q_target,],
                #[I, q, g, oin, I_target, q_target, g_target, label],
                #[I, q, g, I_target, q_target, g_target],
                [train_target, qa],)

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
