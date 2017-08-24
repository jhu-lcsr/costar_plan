from abstract import HierarchicalAgentBasedModel

from robot_multi_models import *
from split import *
from robot_multi_hierarchical import *

class RobotMultiSampler(RobotMultiHierarchical):

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
        super(RobotMultiHierarchical, self).__init__(taskdef, *args, **kwargs)

        self.num_frames = 1

        self.dropout_rate = 0.5
        self.img_dense_size = 1024
        self.img_col_dim = 512
        self.img_num_filters = 128
        self.combined_dense_size = 128
        self.partition_step_size = 2


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

        # Tile on the option -- this is where our transition model comes in.
        # Options are represented as a one-hot vector added to all possible
        # positions in the image, and essentially give us _numLabels()
        # additional image channels.
        tile_width = img_shape[0]/(2**3)
        tile_height = img_shape[1]/(2**3)
        tile_shape = (1, tile_width, tile_height, 1)


        # =====================================================================
        # Add in the chosen option
        option_in = Input((self._numLabels(),),name="chosen_option_in")
        option = Reshape([1,1,self._numLabels()])(option_in)
        option = Lambda(lambda x: K.tile(x, tile_shape))(option)
        enc_with_option = Concatenate(
                axis=-1,
                name="add_option_info")([enc,option])
        enc_with_option = Conv2D(self.img_num_filters,
                kernel_size=[3,3], 
                strides=(1, 1),
                padding='same')(enc_with_option)

        # Append chosen option input -- this is for the high level task
        # dynamics.
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
                            option=self._numLabels(),
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
                            option=self._numLabels(),
                            batchnorm=True,)

        # Predict the next joint states and gripper position. We add these back
        # in from the inputs once again, in order to make sure they don't get
        # lost in all the convolution layers above...
        x = Conv2D(self.img_num_filters/2,
                kernel_size=[5,5], 
                strides=(2, 2),
                padding='same')(enc_with_option)
        x = Flatten()(x)
        x = Concatenate(name="add_current_arm_info")([x, ins[1], ins[2]])
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        arm_out = Dense(arm_size,name="action_arm_goal")(x)
        gripper_out = Dense(gripper_size,name="action_gripper_goal")(x)

        # =====================================================================
        # SUPERVISOR
        # Predict the next option -- does not depend on option
        prev_option_in = Input((self._numLabels(),),name="prev_option_in")
        prev_option = Reshape([1,1,self._numLabels()])(prev_option_in)
        prev_option = Lambda(lambda x: K.tile(x, tile_shape))(prev_option)
        x = Concatenate(axis=-1,name="add_prev_option_to_supervisor")(
                [prev_option, enc])
        for _ in xrange(2):
            # Repeat twice to scale down to a very small size -- this will help
            # a little with the final image layers
            x = Conv2D(self.img_num_filters/4,
                    kernel_size=[5, 5], 
                    strides=(2, 2),
                    padding='same')(x)
            x = Dropout(self.dropout_rate)(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(self.combined_dense_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        label_out = Dense(self._numLabels(), activation="sigmoid")(x)

        supervisor = Model(ins[:3] + [prev_option_in], [label_out])

        enc_with_option_flat = Flatten()(enc_with_option)
        decoder = Model(rep, dec, name="action_image_goal_decoder")
        next_frame_decoder = Model(
                rep2,
                dec2,
                name="action_next_image_decoder")
        features_out = [
                decoder([enc_with_option_flat,option_in]),
                arm_out,
                gripper_out,
                next_frame_decoder([enc_with_option_flat, option_in])]
        predictor = Model(ins, features_out)

        predict_goal = Model(ins, features_out[:3],)
        predict_next = Model(ins, features_out[3])

        return enc, supervisor, predictor, predict_goal, predict_next

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

    def train(self, *args, **kwargs):
        '''
        Pre-process training data.

        Then, create the model. Train based on labeled data. Remove
        unsuccessful examples.
        '''

        # ================================================
        [I, q, g,
                qa,
                ga,
                o_prev,
                oin,
                o_target,
                Inext_target,
                I_target,
                q_target,
                g_target,
                action_labels] = self.preprocess(*args, **kwargs)

        if self.supervisor is None:
            self._makeModel(I, q, g, qa, ga, oin)

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
        self._fitSupervisor([I, q, g, o_prev], o_target)
        # ===============================================

        action_target = [qa, ga]
        self._fitPolicies([I, q, g], action_labels, action_target)
        self._fitBaseline([I, q, g], action_target)

