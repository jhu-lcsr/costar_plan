
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
        
        self.generator_dim = 64
        self.img_dense_size = 512
        self.img_num_filters = 64

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
        opt = self.getOptimizer()
        for inputs, output in zip(ins, outs):
            model = Model(inputs, output)
            model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
            self.models.append(model)
        
        self.generator = self.models[0]
        self.discriminator = self.models[1]

        # =====================================================================
        # Set up adversarial model

        # Create an adversarial version of the model
        self.discriminator.trainable = False
        self.generator.trainable = True
        self.adversarial = Model(
                self.generator.inputs + self.discriminator.inputs[1:],
                self.discriminator([self.generator.outputs[0]] + self.discriminator.inputs[1:])
                )
        self.adversarial.compile(loss=loss, optimizer=self.getOptimizer(),
                metrics=["accuracy"])

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

        # Set up the learning problem as:
        # Goal: f(img, arm, gripper, arm_cmd, gripper_cmd)
        #        --> (img, arm, gripper)
        # At least eventually.

        #print label

        img_shape = features.shape[1:]
        arm_size = arm.shape[1]
        if len(gripper.shape) > 1:
            gripper_size = gripper.shape[1]
        else:
            gripper_size = 1

        enc_ins, enc = GetEncoder(img_shape,
                arm_size,
                gripper_size,
                self.generator_dim,
                self.dropout_rate,
                self.img_num_filters,
                self.img_dense_size,
                discriminator=True)
        dec_ins, dec = GetDecoder(self.generator_dim,
                            img_shape,
                            arm_size,
                            gripper_size,
                            dropout_rate=self.dropout_rate,
                            filters=self.img_num_filters,)

        self.make([dec_ins, enc_ins], [dec, enc], loss="binary_crossentropy")

        if self.show_iter > 0:
            plt.figure()

        self.discriminator.trainable = False
        self.generator.trainable = True
        self.adversarial.summary()

        # pretrain
        print "Pretraining discriminator..."
        self.discriminator.trainable = True
        idx = np.random.randint(0, features.shape[0], size=self.batch_size)
        for i in xrange(self.pretrain_iter):
            # Sample one batch, including random noise
            xi = features[idx]
            ya = arm[idx]
            yg = gripper[idx]
            noise = np.random.random((self.batch_size, self.generator_dim))

            # generate fake data
            data_fake = self.generator.predict([noise])

            # clean up and set up for discriminator batch training
            xi_fake = np.concatenate((xi, data_fake))
            is_fake = np.zeros((2*self.batch_size, 1))
            is_fake[self.batch_size:] = 1
            ya_double = np.concatenate((ya, ya))
            yg_double = np.concatenate((yg, yg))
            d_loss = self.discriminator.train_on_batch(
                    #[xi_fake, ya_double, yg_double],
                    [xi_fake],
                    is_fake)
            print "PT Iter %d: pretraining discriminator loss = "%(i+1), d_loss

        self.discriminator.trainable = False

        for i in xrange(self.iter):

            # Sample one batch, including random noise
            idx = np.random.randint(0, features.shape[0], size=self.batch_size)
            xi = features[idx]
            ya = arm[idx]
            yg = gripper[idx]
            noise = np.random.random((self.batch_size, self.generator_dim))

            # Sample fake data
            data_fake = self.generator.predict([noise])
            #data_fake = self.generator.predict([noise])

            # Train discriminator
            self.discriminator.trainable = True
            xi_fake = np.concatenate((xi, data_fake))
            is_fake = np.zeros((2*self.batch_size, 1))
            is_fake[self.batch_size:] = 1
            ya_double = np.concatenate((ya, ya))
            yg_double = np.concatenate((yg, yg))

            #d_loss = self.discriminator.train_on_batch([xi_fake, ya_double,
            #    yg_double], is_fake)
            d_loss = self.discriminator.train_on_batch([xi_fake], is_fake)
            self.discriminator.trainable = False

            self.generator.trainable = True
            # resample noise
            idx = np.random.randint(0, features.shape[0], size=self.batch_size)
            noise = np.random.random((self.batch_size, self.generator_dim))
            fake = np.zeros((self.batch_size, 1))
            g_loss = self.adversarial.train_on_batch(
                    #[noise, ya, yg],
                    [noise],
                    fake,)
            self.generator.trainable = False

            #print "actual loss", np.mean(np.sum(np.square(p - fake))), p
            print "Iter %d: D loss / GAN loss = "%(i+1), d_loss, g_loss

            if self.show_iter > 0 and (i + 1) % self.show_iter == 0:
                for j in xrange(6):
                    plt.subplot(2, 3, j+1)
                    plt.imshow(np.squeeze(data_fake[j]))
                    plt.axis('off')
                    plt.tight_layout()
                plt.ion()
                plt.show(block=False)
                plt.pause(0.001)

    def save(self):
        '''
        Save to a filename determined by the "self.name" field. In this case we
        save multiple files for the different models we learned.
        '''
        if self.adversarial is not None:
            self.adversarial.save_weights(self.name + "_adversarial.h5f")
            self.discriminator.save_weights(self.name + "_discriminator.h5f")
            self.generator.save_weights(self.name + "_generator.h5f")
        else:
            raise RuntimeError('save() failed: model not found.')

    def load(self):
        '''
        Load will use the current model descriptor and name to load the file
        that you are interested in, at least for now.
        '''
        raise NotImplementedError('load() not supported yet.')
