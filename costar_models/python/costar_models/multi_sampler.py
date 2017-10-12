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
from .multi_hierarchical import *
from .robot_multi_models import *
from .split import *
from .mhp_loss import *
from .loss import *

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
        self.img_num_filters = 64
        self.tform_filters = 64
        self.num_hypotheses = 4
        self.validation_split = 0.1
        self.num_options = 48
        self.num_features = 4

        self.extra_layers = 1
        self.steps_down = 4
        self.num_actor_policy_layers = 2
        self.num_generator_layers = 1
        self.num_arm_vars = 6

        # Number of nonlinear transformations to be applied to the hidden state
        # in order to compute a possible next state.
        if self.dense_representation:
            self.num_transforms = 2
        else:
            self.num_transforms = 3

        # Feature presets
        self.arm_cmd_size = 6
        self.gripper_cmd_size = 1

        # Used for classifiers: value and next option
        self.combined_dense_size = 128

        # Size of the "pose" column containing arm, gripper info
        self.pose_col_dim = 64

        # Size of the hidden representation when using dense
        self.img_col_dim = 128

        self.PredictorCb = PredictorShowImage
        self.hidden_dim = 64/(2**self.steps_down)
        self.hidden_shape = (self.hidden_dim,self.hidden_dim,self.tform_filters)

        self.predictor = None
        self.train_predictor = None
        self.actor = None

        # ===================================================================
        # These are hard coded settings -- tweaking them may break a bunch of
        # things.
        self.use_prev_option = True
        self.always_same_transform = False

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
        image_size = 1
        for dim in img_shape:
            image_size *= dim
        image_size = int(image_size)    

        ins, enc, skips, robot_skip = GetEncoder(img_shape,
                [arm_size, gripper_size],
                self.img_col_dim,
                dropout_rate=self.dropout_rate,
                filters=self.img_num_filters,
                leaky=False,
                dropout=True,
                pre_tiling_layers=self.extra_layers,
                post_tiling_layers=self.steps_down,
                stride1_post_tiling_layers=1,
                pose_col_dim=self.pose_col_dim,
                kernel_size=[5,5],
                dense=self.dense_representation,
                batchnorm=True,
                tile=True,
                flatten=False,
                option=self.num_options,
                #option=None,
                output_filters=self.tform_filters,
                )
        img_in, arm_in, gripper_in, option_in = ins
        #img_in, arm_in, gripper_in = ins
        if self.use_noise:
            z = Input((self.num_hypotheses, self.noise_dim))

        # =====================================================================
        # Create the decoders for image, arm, gripper.
        decoder = GetImageArmGripperDecoder(
                        self.img_col_dim,
                        img_shape,
                        dropout_rate=self.decoder_dropout_rate,
                        dense_size=self.combined_dense_size,
                        kernel_size=[3,3],
                        filters=self.img_num_filters,
                        stride2_layers=self.steps_down,
                        stride1_layers=self.extra_layers,
                        tform_filters=self.tform_filters,
                        num_options=self.num_options,
                        arm_size=arm_size,
                        gripper_size=gripper_size,
                        dropout=self.hypothesis_dropout,
                        upsampling=self.upsampling_method,
                        leaky=True,
                        dense=self.dense_representation,
                        dense_rep_size=self.img_col_dim,
                        skips=self.skip_connections,
                        robot_skip=robot_skip,
                        resnet_blocks=self.residual,
                        batchnorm=True,)


        image_outs = []
        arm_outs = []
        gripper_outs = []
        train_outs = []
        label_outs = []

        if self.skip_connections:
            skips.reverse()
        decoder.compile(loss="mae",optimizer=self.getOptimizer())
        decoder.summary()
    
        #option_in = Input((1,),name="prev_option_in")
        #ins += [option_in]
        value_out, next_option_out = GetNextOptionAndValue(enc,
                                                           self.num_options,
                                                           option_in=None)
                                                           #option_in=option_in)

        # =====================================================================
        # Create many different image decoders
        if self.always_same_transform:
            transform = self._getTransform(0)
        for i in range(self.num_hypotheses):
            if not self.always_same_transform:
                transform = self._getTransform(i)

            if i == 0:
                transform.summary()
            if self.use_noise:
                zi = Lambda(lambda x: x[:,i], name="slice_z%d"%i)(z)
                #x = transform([enc, zi, next_option_out])
                x = transform([enc, zi])
            else:
                x = transform([enc])
            
            # This maps from our latent world state back into observable images.
            if self.skip_connections:
                decoder_inputs = [x] + skips
            else:
                decoder_inputs = [x]
            img_x, arm_x, gripper_x, label_x = decoder(decoder_inputs)

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
        predictor = Model(ins + [z],
                [image_out, arm_out, gripper_out, label_out, next_option_out,
                    value_out])
        actor = None
        train_predictor = Model(ins + [z],
                [train_out, next_option_out, value_out])

        # =====================================================================
        # Create models to train
        train_predictor.compile(
                loss=[#"mae","mse","mse","binary_crossentropy",
                    MhpLossWithShape(
                        num_hypotheses=self.num_hypotheses,
                        outputs=[image_size, arm_size, gripper_size, self.num_options],
                        weights=[0.7,0.3,0.1,0.1],
                        loss=["mae","mse","mse","categorical_crossentropy"],
                        avg_weight=0.05),
                    "binary_crossentropy", "binary_crossentropy"],
                loss_weights=[#0.1,0.1,0.1,0.1,
                    1.0,0.01,0.01],
                optimizer=self.getOptimizer())
        predictor.compile(loss="mae", optimizer=self.getOptimizer())
        train_predictor.summary()

        return predictor, train_predictor, actor

    def _getTransform(self,i=0):
        transform_dropout = False
        use_options_again = False
        if use_options_again:
            options = self.num_options
        else:
            options = None
        if self.dense_representation:
            transform = GetDenseTransform(
                    dim=self.img_col_dim,
                    input_size=self.img_col_dim,
                    output_size=self.img_col_dim,
                    idx=i,
                    batchnorm=True,
                    dropout=transform_dropout,
                    dropout_rate=self.dropout_rate,
                    leaky=True,
                    num_blocks=self.num_transforms,
                    relu=True,
                    option=options,
                    resnet_blocks=self.residual,
                    use_noise=self.use_noise,
                    noise_dim=self.noise_dim,)
        else:
            transform = GetTransform(
                    rep_size=(self.hidden_dim, self.hidden_dim),
                    filters=self.tform_filters,
                    kernel_size=[3,3],
                    idx=i,
                    batchnorm=True,
                    dropout=transform_dropout,
                    dropout_rate=self.dropout_rate,
                    leaky=True,
                    num_blocks=self.num_transforms,
                    relu=True,
                    option=options,
                    resnet_blocks=self.residual,
                    use_noise=self.use_noise,
                    noise_dim=self.noise_dim,)
        return transform

    def _makeModel(self, features, arm, gripper, *args, **kwargs):
        '''
        Little helper function wraps makePredictor to consturct all the models.

        Parameters:
        -----------
        features, arm, gripper: variables of the appropriate sizes
        '''
        self.predictor, self.train_predictor, self.actor = \
            self._makePredictor(
                (features, arm, gripper))

    def _makeTrainTarget(self, I_target, q_target, g_target, o_target):
        length = I_target.shape[0]
        image_shape = I_target.shape[1:]
        image_size = 1
        for dim in image_shape:
            image_size *= dim
        image_size = int(image_size)
        Itrain = np.reshape(I_target,(length, image_size))
        return np.concatenate([Itrain, q_target,g_target,o_target],axis=-1)

    def _getAllData(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            prev_label, goal_features, goal_arm, goal_gripper, value, *args, **kwargs):
        I = features
        q = arm
        g = gripper * -1
        qa = arm_cmd
        ga = gripper_cmd * -1
        oin = prev_label
        I_target = goal_features
        q_target = goal_arm
        g_target = goal_gripper * -1
        o_target = label
        value_target = np.array(value > 1.,dtype=float)

        o_target = np.squeeze(self.toOneHot2D(o_target, self.num_options))
        train_target = self._makeTrainTarget(
                I_target,
                q_target,
                g_target,
                o_target)

        return [I, q, g, oin, q_target, g_target,], [
                np.expand_dims(train_target, axis=1),
                o_target,
                value_target,
                np.expand_dims(qa, axis=1),
                np.expand_dims(ga, axis=1),
                I_target]


    def _getData(self, *args, **kwargs):
        features, targets = self._getAllData(*args, **kwargs)
        tt, o1, v, qa, ga, I = targets
        if self.use_noise:
            noise_len = features[0].shape[0]
            z = np.random.random(size=(noise_len,self.num_hypotheses,self.noise_dim))
            return features[:self.num_features] + [z], [tt, o1, v]
        else:
            return features[:self.num_features], [tt, o1, v]

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

        [I, q, g, oprev, q_target, g_target,] = features
        [I_target2, o_target, value_target, qa, ga, I_target0] = targets

        if self.predictor is None:
            self._makeModel(I, q, g, qa, ga)

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
        #assert train_size == 12296 + self.num_options
        assert train_size == (64*64*3) + self.num_arm_vars + 1 + self.num_options

        # ===================================================================
        # Create the callbacks and actually run the training loop.
        modelCheckpointCb = ModelCheckpoint(
            filepath=self.name+"_predictor_weights.h5f",
            verbose=1,
            save_best_only=True # does not work without validation wts
        )
        imageCb = self.PredictorCb(
            self.predictor,
            features=features[:4],
            targets=targets,
            model_directory=self.model_directory,
            num_hypotheses=self.num_hypotheses,
            verbose=True,
            use_noise=self.use_noise,
            noise_dim=self.noise_dim,
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
                print("Could not load actor:", e)
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

    def _makeActorPolicy(self):
        '''
        Helper function: creates a model for the "actor" policy that will
        generate the controls to move towards a particular end effector pose.
        The job of this policy should be pretty simple.

        The actor policy is trained separately from the predictor/sampler
        policies, but using the same underlying representation.
        '''
        enc = Input((self.img_col_dim,))
        arm_goal = Input((self.num_arm_vars,),name="actor_arm_goal_in")
        gripper_goal = Input((1,),name="actor_gripper_goal_in")
        y = enc
        if not self.dense_representation:
            raise RuntimeError('Not yet supported!')
            y = Conv2D(int(self.img_num_filters/4),
                    kernel_size=[5,5], 
                    strides=(2, 2),
                    padding='same')(y)
            y = Dropout(self.dropout_rate)(y)
            y = LeakyReLU(0.2)(y)
            y = BatchNormalization(momentum=0.9)(y)
            y = Flatten()(y)
        else:
            y = Concatenate()([y, arm_goal, gripper_goal])
            for _ in range(self.num_actor_policy_layers):
                y = Dense(self.combined_dense_size)(y)
                y = BatchNormalization(momentum=0.9)(y)
                y = LeakyReLU(0.2)(y)
                y = Dropout(self.dropout_rate)(y)
        arm_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="arm_action")(
                Dense(self.arm_cmd_size)(y))
        gripper_cmd_out = Lambda(lambda x: K.expand_dims(x, axis=1),name="gripper_action")(
                Dense(self.gripper_cmd_size)(y))
        actor = Model([enc, arm_goal, gripper_goal], [arm_cmd_out,
            gripper_cmd_out], name="actor")
        return actor

    def _makeGenerator(self, img_shape, kernel_size, skips=None):
        rep, dec = GetImageDecoder(self.img_col_dim,
                        img_shape,
                        dropout_rate=self.dropout_rate,
                        kernel_size=kernel_size,
                        filters=self.img_num_filters,
                        stride2_layers=self.steps_down,
                        stride1_layers=self.extra_layers,
                        tform_filters=self.tform_filters,
                        dropout=self.hypothesis_dropout,
                        upsampling=self.upsampling_method,
                        dense=self.dense_representation,
                        dense_rep_size=self.img_col_dim,
                        leaky=True,
                        skips=skips,
                        original=None,
                        resnet_blocks=self.residual,
                        batchnorm=True,)
        decoder = Model(rep, dec)
        decoder.compile(loss="mae",optimizer=self.getOptimizer())
        hidden = Input((self.img_col_dim,), name="generator_hidden_in")
        arm_goal = Input((self.num_arm_vars,),name="generator_arm_goal_in")
        gripper_goal = Input((1,),name="generator_gripper_goal_in")

        y = Concatenate()([hidden, arm_goal, gripper_goal])
        for _ in range(self.num_generator_layers):
            y = Dense(self.img_col_dim)(y)
            y = BatchNormalization(momentum=0.9)(y)
            y = LeakyReLU(0.2)(y)
            y = Dropout(self.dropout_rate)(y)
    
        skip_ins = []
        if self.skip_connections:
            for skip in skips:
                shape = [int(i) for i in skip.shape[1:]]
                skip_ins.append(Input(shape))
            img_out = decoder([y] + skip_ins)
        else:
            img_out = decoder([y])

        generator = Model(
                [hidden, arm_goal, gripper_goal] + skip_ins,
                img_out,
                name="generator")
        return generator

    def _makeImageDecoder(self, img_shape, kernel_size, skips=None):
        rep, dec = GetImageDecoder(self.img_col_dim,
                        img_shape,
                        dropout_rate=self.dropout_rate,
                        kernel_size=kernel_size,
                        filters=self.img_num_filters,
                        stride2_layers=self.steps_down,
                        stride1_layers=self.extra_layers,
                        tform_filters=self.tform_filters,
                        dropout=self.hypothesis_dropout,
                        upsampling=self.upsampling_method,
                        dense=self.dense_representation,
                        dense_rep_size=self.img_col_dim,
                        leaky=True,
                        skips=skips,
                        original=None,
                        resnet_blocks=self.residual,
                        batchnorm=True,)
        decoder = Model(rep, dec, name="image_decoder")
        decoder.compile(loss="mae",optimizer=self.getOptimizer())
        return decoder
