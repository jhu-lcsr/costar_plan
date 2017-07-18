
import keras.backend as K
import keras.losses as losses
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

from abstract import AbstractAgentBasedModel
from gan import GAN

from robot_multi_models import *

class RobotMultiGAN(AbstractAgentBasedModel):
    '''
    This is a version of the GAN agent based model. It doesn't really inherit
    too much from that though.

    This model is designed to work with the "--features multi" option of the
    costar bullet sim. This includes multiple viewpoints of a scene, including
    camera input and others.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        """
        Read in taskdef for this model (or set of models). Use it to create the
        generator and discriminator neural nets that we will be optimizing.
        """

        self.taskdef = taskdef
        
        img_rows = 768 / 8
        img_cols = 1024 / 8
        self.nchannels = 3

        self.img_shape = (img_rows, img_cols, self.nchannels)

        self.generator_dim = 1024
        self.img_dense_size = 1024
        self.img_num_filters =64

        self.dropout_rate = 0.5

        #self.generator_dense_size = 1024
        #self.generator_filters_c1 = 256
        #self.discriminator_dense_size = 1024
        #self.discriminator_filters_c1 = 512


        self.adversarial = None
        self.generator = None
        self.discriminator = None
        self.models = []

        super(RobotMultiGAN, self).__init__(*args, **kwargs)

    '''
    Make the adversarial model and prepare for training.

    Parameters:
    -----------
    ins: lists of input variables for each model. List of lists of tensors.
    outs: output variable for each model. List of tensors.
    ops: optimizers associated with each output.
    loss: loss function associated with each output.
    noise dim: how much noise we generate as a vector to seed various samples.
    '''
    def make(self, ins, outs, loss,):

        # =====================================================================
        # Compile all the basic models
        for inputs, output, opt in zip(ins, outs, opts):
            model = Model(inputs, output)
            model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
            self.models.append(model)
        #self.models = outs
        #for model, opt in zip(self.models, opts):
        #    model.compile(optimizer=opt, loss=loss)
        #ins[0] = self.models[0].inputs[0]
        
        self.generator = self.models[0]
        self.discriminator = self.models[1]

        # =====================================================================
        # Set up adversarial model

        # Create an adversarial version of the model
        self.discriminator.trainable = False
        self.adversarial = Model(
                ins[0],
                self.discriminator([self.generator.outputs[0],
                    self.discriminator.inputs[1:]])
                )
        self.adversarial.compile(loss=loss, optimizer=getOptimizer())
        self.summary()

    def train(self, features, arm, gripper, arm_cmd, gripper_cmd, label,
            example, *args, **kwargs):
        '''
        Set up the imitation GAN to learn a model of what actions we expect
        from each state. Our goal is to sample the distribution of actions that
        is most likely to give us a trajectory to the goal.
        '''
        """
        imgs = data['features']
        arm = data['arm']
        gripper = data['gripper']
        arm_cmd = data['arm_cmd']
        gripper_cmd = data['gripper_cmd']
        labels = data['action']
        """

        print label

        # Set up the learning problem as:
        # Goal: f(img, arm, gripper) --> arm_cmd, gripper_cmd

        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1

        enc_ins, enc = GetEncoder(img_shape,
                arm_size,
                gripper_size,
                self.img_col_dim,
                self.dropout_rate,
                self.img_num_filters,
                self.img_dense_size,
                discriminator=True)
        dec_ins, dec = GetDecoder(self.img_dense_size,
                            img_shape,
                            arm_size,
                            gripper_size,
                            self.dropout_rate,
                            self.generator_filters_c1,
                            self.robot_dense_size)

        self.make([enc_ins, dec_ins], [enc, dec])

        for i in xrange(num_iter):

            # Sample one batch, including random noise
            idx = np.random.randint(0, y.shape[0], size=batch_size)
            xi = x[idx]
            yf = features[idx]
            ya = arm[idx]
            noise = np.random.random((batch_size, self.noise_dim))

            # Sample fake data
            data_fake = self.generator.predict([noise, yi])
            #data_fake = self.generator.predict([noise])

            # Train discriminator
            self.discriminator.trainable = True
            xi_fake = np.concatenate((xi, data_fake))
            is_fake = np.zeros((2*batch_size, 1))
            is_fake[batch_size:] = 1
            ya_double = np.concatenate((ya, ya))
            yg_double = np.concatenate((yg, yg))
            d_loss = self.discriminator.train_on_batch([xi_fake, ya_double,
                yg_double], is_fake)

            g_loss = self.adversarial.train_on_batch(
                    [noise, yi],
                    #[noise],
                    np.zeros((batch_size, 1)),
                            )

            print "Iter %d: D loss / GAN loss = "%(i), d_loss, g_loss

            if (i + 1) % 25 == 0:
                for j in xrange(6):
                    plt.subplot(2, 3, j+1)
                    plt.imshow(np.squeeze(data_fake[j]), cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.show(block=False)
    def fi
