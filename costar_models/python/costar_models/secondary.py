from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from .callbacks import *
from .sampler2 import *
from .data_utils import GetNextGoal, ToOneHot2D
from .multi import *


class Secondary(PredictionSampler2):
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
        super(Secondary, self).__init__(*args, **kwargs)
        self.PredictorCb = None
        self.save_encoder_decoder = False
        self.save_train_state = False

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
        arm_in = Input((arm_size,))
        gripper_in = Input((gripper_size,))
        next_option_in = Input((1,), name="next_option_in")
        next_option_in2 = Input((1,), name="next_option_in2")
        label_in = Input((1,))
        ins = [img0_in, img_in, arm_in, gripper_in, next_option_in,
                next_option_in2, label_in]

        encoder = self._makeImageEncoder(img_shape)
        decoder = self._makeImageDecoder(self.hidden_shape)
        LoadEncoderWeights(self, encoder, decoder)

        # =====================================================================
        # Load the arm and gripper representation
        h = encoder([img_in])
        h0 = encoder(img0_in)

        # ====================================================================
        # Compute the option
        y = OneHot(self.num_options)(next_option_in)
        y = Flatten()(y)

        actor = None
        if self.submodel == "value":
            model = GetValueModel(h, self.num_options, 128,
                    self.decoder_dropout_rate)
            model.compile(loss="mae", optimizer=self.getOptimizer())
            self.value_model = model
            outs = model([h])
            loss = "binary_crossentropy"
            loss_wts = [1,]
            metrics=["accuracy"]
        elif self.submodel == "q":
            model = GetNextModel(h, self.num_options, 128,
                    self.decoder_dropout_rate)
            model.compile(loss="mae", optimizer=self.getOptimizer())
            outs = model([h0,h,label_in])
            self.q_model = model
            loss = "binary_crossentropy"
            loss_wts = [1,1]
            metrics=["accuracy"]
        elif self.submodel == "next":
            model = GetNextModel(h, self.num_options, 128,
                    self.decoder_dropout_rate)
            model.compile(loss="mae", optimizer=self.getOptimizer())
            outs = model([h0,h,label_in])
            self.next_model = model
            loss = "binary_crossentropy"
            loss_wts = [1,1]
            metrics=["accuracy"]
        elif self.submodel == "actor":
            actor = GetActorModel(h, self.num_options, arm_size, gripper_size,
                    self.decoder_dropout_rate)
            actor.compile(loss="mae",optimizer=self.getOptimizer())
            model = actor
            outs = actor([h0, h, arm_in, gripper_in, y])
            loss = self.loss
            loss_wts = [1,0.2]
            metrics=[]
        elif self.submodel == "pose":
            model = GetPoseModel(h, self.num_options, arm_size, gripper_size,
                    self.decoder_dropout_rate)
            model.compile(loss="mae",optimizer=self.getOptimizer())
            self.pose_model = model
            outs = model([h0, h, y, arm_in, gripper_in])
            loss = self.loss
            loss_wts = [1,0.2]
            metrics=[]

        model.summary()
        # =====================================================================
        train_predictor = Model(ins, outs)
        train_predictor.compile(loss=loss,
                loss_weights=loss_wts,
                metrics=metrics,
                optimizer=self.getOptimizer())
        return None, train_predictor, actor, ins, h

    def _getData(self, *args, **kwargs):
        features, targets = GetAllMultiData(self.num_options, *args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        I_target2, o2 = GetNextGoal(I_target, o1)
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        oin_1h = np.squeeze(ToOneHot2D(oin, self.num_options))
        o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
        o2_1h = np.squeeze(ToOneHot2D(o2, self.num_options))

        qa = np.squeeze(qa)
        ga = np.squeeze(ga)
        o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
        done = np.ones_like(oin) - (oin == o1)
        if self.submodel == "value":
            outs = [v]
        elif self.submodel == "next":
            outs = [o1_1h, done]
        elif self.submodel == "q":
            if len(v.shape) == 1:
                v = np.expand_dims(v,axis=1)
            vs = np.repeat(v, self.num_options, axis=1)
            outs = [o1_1h * vs, done]
        elif self.submodel == "actor":
            outs = [qa, ga]
        elif self.submodel == "pose":
            outs = [q_target, g_target]
        return ([I0, I, q, g, o1, o2, oin], outs)

from .husky import *
class HuskySecondary(Secondary):

    def __init__(self, *args, **kwargs):
        super(HuskySecondary, self).__init__(*args, **kwargs)
        self.num_options = HuskyNumOptions()
        self.null_option = HuskyNullOption()

    def _makeModel(self, image, pose, action, *args, **kwargs):
       
        img_shape = image.shape[1:]
        pose_size = pose.shape[-1]
        action_size = action.shape[-1]

       # =====================================================================
        # Load the image decoders
        img_in = Input(img_shape,name="predictor_img_in")
        img0_in = Input(img_shape,name="predictor_img0_in")
        label_in = Input((1,))
        pose_in = Input((pose_size,))

        encoder = self._makeImageEncoder(img_shape)
        decoder = self._makeImageDecoder(self.hidden_shape)

        LoadEncoderWeights(self, encoder, decoder, gan=False)

        # =====================================================================
        # Load the arm and gripper representation
        h = encoder([img_in])
        h0 = encoder(img0_in)

        next_option_in = Input((1,), name="next_option_in")
        next_option_in2 = Input((1,), name="next_option_in2")
        ins = [img0_in, img_in, pose_in,
               next_option_in, next_option_in2,
               label_in]
        y = OneHot(self.num_options)(next_option_in)
        y = Flatten()(y)

        actor = None
        loss_wts = [1.]
        if self.submodel == "value":
            model = GetValueModel(h, self.num_options, 128,
                    self.decoder_dropout_rate)
            model.compile(loss="mae", optimizer=self.getOptimizer())
            self.value_model = model
            outs = model([h])
            loss = "binary_crossentropy"
            metrics=["accuracy"]
        elif self.submodel == "q":
            model = GetNextModel(h, self.num_options, 128,
                    self.decoder_dropout_rate)
            model.compile(loss="mae", optimizer=self.getOptimizer())
            outs = model([h0,h,label_in])
            self.q_model = model
            loss = "binary_crossentropy"
            loss_wts = [1,1]
            metrics=["accuracy"]
        elif self.submodel == "next":
            model = GetNextModel(h, self.num_options, 128,
                    self.decoder_dropout_rate)
            model.compile(loss="mae", optimizer=self.getOptimizer())
            outs = model([h0,h,label_in])
            self.next_model = model
            loss_wts = [1,1]
            loss = "binary_crossentropy"
            metrics=["accuracy"]
        elif self.submodel == "actor":
            actor = GetHuskyActorModel(h, self.num_options, pose_size,
                    self.decoder_dropout_rate)
            actor.compile(loss="mae",optimizer=self.getOptimizer())
            model = actor
            outs = actor([h0, h, y, pose_in])
            loss = self.loss
            loss_wts = [1,0.2,]
            metrics=[]
            self.actor = actor
        elif self.submodel == "pose":
            model = GetHuskyPoseModel(h, self.num_options, pose_size,
                    self.decoder_dropout_rate)
            model.compile(loss="mae",optimizer=self.getOptimizer())
            self.pose_model = model
            outs = model([h0, h, y, pose_in])
            loss_wts = [1,0.2,]
            loss = self.loss
            metrics=[]

        model.summary()
        # =====================================================================
        train_predictor = Model(ins, outs)
        train_predictor.compile(loss=loss,
                loss_weights=loss_wts,
                metrics=metrics,
                optimizer=self.getOptimizer())

        self.model = train_predictor

    def _getData(self, image, pose, action, label, example,
        prev_label, goal_image, goal_pose, value, reward, *args, **kwargs):

        # --------------------------------------
        # Process the data
        I = np.array(image) / 255.
        p = np.array(pose)
        p_target = np.array(goal_pose)
        a = np.array(action)
        I_target = np.array(goal_image) / 255.
        q_target = np.array(goal_pose)
        oin = np.array(prev_label)
        o1 = np.array(label)
        #v = np.array(np.array(value) > 1.,dtype=float)
        v = np.ones_like(o1)
        success = 1. if np.sum(reward) > 0. else 0.
        v = v * float(success)

        # -------------------------------------
        I_target2, o2 = GetNextGoal(I_target, o1)
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        oin_1h = np.squeeze(ToOneHot2D(oin, self.num_options))
        o1_1h = np.squeeze(ToOneHot2D(o1, self.num_options))
        o2_1h = np.squeeze(ToOneHot2D(o2, self.num_options))
        p_target = np.squeeze(p)
        a = np.squeeze(a)
        # There should probably be an assert here
       	#print(label, example)

        done = np.ones_like(oin) - (oin == o1)

        if self.submodel == "value":
            outs = [v]
        elif self.submodel == "next":
            outs = [o1_1h, done]
        elif self.submodel == "q":
            if len(v.shape) == 1:
                v = np.expand_dims(v,axis=1)
            vs = np.repeat(v, self.num_options, axis=1)
            outs = [o1_1h * vs, done]
        elif self.submodel == "actor":
            outs = [a]
        elif self.submodel == "pose":
            outs = [p_target]
        return ([I0, I, p, o1, o2, oin], outs)
