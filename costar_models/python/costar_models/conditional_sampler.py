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

from .multi import *
from .planner import *
from .conditional_image import ConditionalImage

class ConditionalSampler(ConditionalImage):
    '''
    Version of the sampler that only produces results conditioned on a
    particular action; this version does not bother trying to learn a separate
    distribution for each possible state.
    '''

    def __init__(self, *args, **kwargs):
        '''
        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.

        Parameters:
        -----------
        taskdef: definition of the problem used to create a task model
        '''
        super(ConditionalSampler, self).__init__(*args, **kwargs)

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
        #arm_in = Input((arm_size,))
        #gripper_in = Input((gripper_size,))
        label_in = Input((1,))
        #ins = [img0_in, img_in, arm_in, gripper_in]
        ins = [img0_in, img_in]#, arm_in, gripper_in]

        encoder = MakeImageEncoder(self, img_shape)
        decoder = MakeImageDecoder(self, self.hidden_shape)

        # =====================================================================
        # Encode initial state information
        h = encoder([img0_in, img_in])

        next_option_in = Input((1,), name="next_option_in")
        next_option_in2 = Input((1,), name="next_option_in2")
        ins += [next_option_in, next_option_in2]

        # =====================================================================
        # Create ancillary models
        V = GetValueModel(h, self.num_options, 128,
                          self.decoder_dropout_rate)
        Q = GetNextModel(h, self.num_options, 128,
                    self.decoder_dropout_rate, name="Q", add_done=True)
        next = GetNextModel(h, self.num_options, 128,
                    self.decoder_dropout_rate, name="next", add_done=False)
        self.value_model = V
        self.q_model = Q
        self.next_model = next



        # =====================================================================
        # Apply transforms
        y = Flatten()(OneHot(self.num_options)(next_option_in))
        y2 = Flatten()(OneHot(self.num_options)(next_option_in2))

        tform = self._makeTransform() if not self.dense_transform else self._makeDenseTransform()
        x = tform([h,y])
        x2 = tform([x,y2])

        image_out, image_out2 = decoder([x]), decoder([x2])

        # Compute classifier on the last transform
        if not self.no_disc:
            image_discriminator = LoadGoalClassifierWeights(self,
                    make_classifier_fn=MakeImageClassifier,
                    img_shape=img_shape)
            #disc_out1 = image_discriminator([img0_in, image_out])
            disc_out2 = image_discriminator([img0_in, image_out2])

        # Compute ancillary outputs
        v_out = V(h)
        q_out, q_done_out = Q([h, label_in])
        next_out = next([h, label_in])

        # Set weights for different model terms
        img_loss_wt = 1.
        v_wt = 1e-3
        q_wt = 1e-3
        done_wt = 1e-3
        next_wt = 1e-3

        # Create models to train
        if self.no_disc:
            disc_wt = 0.
        else:
            disc_wt = 1e-3
        ins += [label_in]
        outs = [image_out, image_out2, # image
                next_out,
                v_out,
                q_out,
                q_done_out,
               ]

        losses = [self.loss, self.loss,
                        "binary_crossentropy", # next loss
                        "binary_crossentropy", # value loss
                        "binary_crossentropy", # q loss
                        "binary_crossentropy", # done loss
                        ]
        loss_wts = [img_loss_wt, img_loss_wt,
                next_wt,
                v_wt,
                q_wt,
                done_wt,
                ]

        # If using discriminator
        if not self.no_disc:
            outs += [disc_out2]
            losses += ["categorical_crossentropy"] # discriminator loss
            loss_wts += [disc_wt]

        # Compile model
        train_predictor = Model(ins, outs)
        train_predictor.compile(loss=losses,
                               loss_weights=loss_wts,
                               optimizer=self.getOptimizer())
        train_predictor.summary()
        return None, train_predictor, None, ins, h

    def _getData(self, *args, **kwargs):
        features, targets = GetAllMultiData(self.num_options, *args, **kwargs)
        [I, q, g, oin, label, q_target, g_target,] = features
        tt, o1, v, qa, ga, I_target = targets
        I_target2, o2 = GetNextGoal(I_target, o1)
        I0 = I[0,:,:,:]
        length = I.shape[0]
        I0 = np.tile(np.expand_dims(I0,axis=0),[length,1,1,1]) 
        oin_1h = ToOneHot(oin, self.num_options)
        o1_1h = ToOneHot(o1, self.num_options)
        o2_1h = ToOneHot(o2, self.num_options)
        qa = np.squeeze(qa)
        ga = np.squeeze(ga)
        #features = [I0, I, q, g, o1, o2, oin]
        features = [I0, I, o1, o2, oin]
        done = np.ones_like(oin) - (oin == o1)
        if len(v.shape) == 1:
            vs = np.expand_dims(v,axis=1)
        vs = np.repeat(vs, self.num_options, axis=1)
        qval = o1_1h * vs
        if self.no_disc:
            targets = [I_target, I_target2, o1_1h, v, qval, done,]# qa, ga,]
        else:
            #targets = [I_target, I_target2, o1_1h, v, qval, done, qa, ga, o2_1h]
            targets = [I_target, I_target2, o1_1h, v, qval, done, o2_1h]
            # Uncomment if you want to try the whole "two discriminator" thing
            # again -- this might need a more fully supported option
            #targets = [I_target, I_target2, o1_1h, o2_1h]
            # targets = [I_target, I_target2, o2_1h]
        #if self.enc_loss:
        #    targets += [I_target, I_target2]
        return features, targets


