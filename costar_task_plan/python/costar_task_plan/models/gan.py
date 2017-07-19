
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

__LABELS = False

class GAN(AbstractAgentBasedModel):

    def __init__(self, *args, **kwargs):
        self.models = []
        self.make(*args, **kwargs)

    '''
    This class is designed to wrap some basic functionality for GANs of
    different sorts.

    Parameters:
    -----------
    ins: lists of input variables for each model. List of lists of tensors.
    outs: output variable for each model. List of tensors.
    ops: optimizers associated with each output.
    loss: loss function associated with each output.
    noise dim: how much noise we generate as a vector to seed various samples.
    '''
    def make(self, ins, outs, opts, loss, noise_dim, *args, **kwargs):

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
                #self.discriminator([self.generator.outputs[0]])
                )
        self.adversarial.compile(loss=loss, optimizer=opts[0])
        self.summary()

        """
        adv_loss = losses.get(loss)(self.adversarial.targets[0], self.adversarial.outputs[0])
    
        # Collected trainable weights and sort them deterministically.
        trainable_weights = self.adversarial.trainable_weights

        # Sort weights by name.
        if K.backend() == 'theano':
            trainable_weights.sort(key=lambda x: x.name if x.name else x.auto_name)
        else:
            trainable_weights.sort(key=lambda x: x.name)
 
        updates = \
            self.adversarial.optimizer.get_updates(
                    trainable_weights,
                    self.adversarial.constraints,
                    adv_loss)
        self.optimize_f = K.function(
                inputs=self.adversarial.inputs,
                outputs=self.adversarial.outputs,
                updates=updates)
        """
        #print self.adversarial.losses, self.adversarial.loss_weights, \
        #    self.adversarial.loss
        #print self.adversarial.trainable_weights, \
        #    len(self.adversarial.trainable_weights)
        #print self.generator.trainable_weights, \
        #    len(self.adversarial.trainable_weights)

        self.noise_dim = noise_dim

    def predict(self, label):
        noise = np.random.random((self.noise_dim,))
        return self.generator.predict([noise, label, label])

    def summary(self):
        #print self.generator.summary()
        self.adversarial.summary()
        self.discriminator.summary()

    def fit(self, x, y, num_iter=3001, batch_size=50, save_interval=0):
        for i in xrange(num_iter):

            # Sample one batch, including random noise
            idx = np.random.randint(0, y.shape[0], size=batch_size)
            xi = x[idx]
            yi = y[idx]
            noise = np.random.random((batch_size, self.noise_dim))

            # Sample fake data
            data_fake = self.generator.predict([noise, yi])
            #data_fake = self.generator.predict([noise])

            # Train discriminator
            self.discriminator.trainable = True
            xi_fake = np.concatenate((xi, data_fake))
            is_fake = np.zeros((2*batch_size, 1))
            is_fake[batch_size:] = 1
            yi_double = np.concatenate((yi, yi))
            d_loss = self.discriminator.train_on_batch([xi_fake, yi_double], is_fake)
            self.discriminator.trainable = False

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

    def train(self, data, *args, **kwargs):
        raise NotImplementedError('No train function set up.')

    def _adversarial(self):
        NotImplementedError('return adversarial model')

    def _discriminator(self):
        NotImplementedError('return discriminator model')

    def _generator(self):
        NotImplementedError('return generator model')

class SimpleImageGan(GAN):
    '''
    This is designed specifically for MNIST, but could be used in principle
    with plenty of other data sets. Or at least something pretty similar could
    be.

    - generator: takes as input generator_seed, outputs image of size
      image_shape.
    - discriminator: model of size 
    '''

    def __init__(self, img_rows=28, img_cols=28, channels=1, labels=10,
            noise_dim=100,):
        self.img_shape = (img_rows, img_cols, channels)

        self.generator_dense_size = 1024
        self.generator_filters_c1 = 256

        self.discriminator_dense_size = 1024
        self.discriminator_filters_c1 = 512

        self.dropout_rate = 0.5

        g_in, g_out, g_opt = self._generator(self.img_shape, labels, noise_dim)
        labels_input = g_in[-1]
        d_in, d_out, d_opt = self._discriminator(self.img_shape, labels,
                labels_input)
        #g_in, g_out, g_opt = self.model_generator()
        #d_in, d_out, d_opt = self.model_discriminator()

        super(SimpleImageGan, self).__init__(
                [g_in, d_in],
                [g_out, d_out],
                [g_opt, d_opt],
                "binary_crossentropy",
                noise_dim)

    def _generator(self, img_shape, num_labels, noise_dim):
        height4 = img_shape[0]/4
        width4 = img_shape[1]/4
        height2 = img_shape[0]/2
        width2 = img_shape[1]/2

        labels = Input(shape=(num_labels,))
        noise = Input(shape=(noise_dim,))

        x = Concatenate(axis=1)([noise, labels])
        
        # Add first dense layer
        #x = Dense(self.generator_dense_size)(noise)
        cnn_inputs = self.generator_filters_c1
        #x = Dense(cnn_inputs * height4 * width4)(noise)
        x = Dense(cnn_inputs * height2 * width2)(x)
        #x = Dense(cnn_inputs * height2 * width2)(noise)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Add labels and adjust size -- append to every space
        #labels2 = RepeatVector(height2*width2)(labels)
        #labels2 = Reshape((height4,width4,num_labels))(labels2)
        #labels2 = Reshape((height2,width2,num_labels))(labels2)
        assert not K.image_dim_ordering() == 'th'
        #x = Reshape((height4, width4, cnn_inputs))(x)
        x = Reshape((height2, width2, cnn_inputs))(x)
        #x = Concatenate(axis=3)([x,labels2])

        # Apply the first convolution
        #x = Conv2DTranspose(cnn_inputs,
        #           kernel_size=[5, 5], 
        #           strides=(2, 2),
        #           padding="same")(x)
        #x = BatchNormalization(axis=-1, scale=False, center=False, momentum=0.9, epsilon=1e-5)(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(int(cnn_inputs / 2), 3, 3, border_mode='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)

        # Add labels in again -- to each column
        labels2 = RepeatVector(height2*width2)(labels)
        labels2 = Reshape((height2,width2,num_labels))(labels2)
        #x = Concatenate(axis=3)([x,labels2])

        # Second convolution
        #x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(int(cnn_inputs / 4), 3, 3, border_mode='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)
        
        # Final convolution
        x = Conv2D(1, 1, 1, border_mode='same')(x)

        # Apply the second convolution
        #x = Conv2DTranspose(self.img_shape[2],
        #           kernel_size=[5, 5], 
        #           strides=(2, 2),
        #           padding="same")(x)
        #x = BatchNormalization(axis=-1, scale=False, center=False, momentum=0.9, epsilon=1e-5)(x)
        x = Activation('sigmoid')(x)

        generator_optimizer = Adam(lr=2e-4, beta_1=0.5)

        return [noise, labels], x, generator_optimizer

    def _discriminator(self, img_shape, num_labels, labels):
        # just for my sanity
        height4 = img_shape[0]/4
        width4 = img_shape[1]/4
        height2 = img_shape[0]/2
        width2 = img_shape[1]/2
        height = img_shape[0]
        width = img_shape[1]
        channels = img_shape[2]

        # sampled images
        samples = Input(shape=img_shape)

        # labels created, copied, and reshaped
        #labels2 = RepeatVector(height*width)(labels)
        #labels2 = Reshape((height,width,num_labels))(labels2)

        # Add initial discriminator layer
        #x = Concatenate(axis=3)([samples, labels2])
        x = Conv2D(int(self.discriminator_filters_c1 / 2), # + num_labels
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   #padding="same")(x)
                   padding="same")(samples)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.dropout_rate)(x)

        # Add conv layer with more filters
        #labels2 = RepeatVector(height2*width2)(labels)
        #labels2 = Reshape((height2,width2,num_labels))(labels2)
        #x = Concatenate(axis=3)([x, labels2])
        x = Conv2D(self.discriminator_filters_c1, # + num_labels
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   padding="same")(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.dropout_rate)(x)

        # Add dense layer
        x = Flatten()(x)
        x = Concatenate(axis=1)([x, labels])
        x = Dense(int(0.5 * self.discriminator_filters_c1))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.dropout_rate)(x)

        # Single output -- sigmoid activation function
        x = Concatenate(axis=1)([x, labels])
        x = Dense(1, activation='sigmoid')(x)

        discriminator_optimizer = Adam(lr=1e-4, beta_1=0.5)
        #x = Model([samples, labels], x)

        return [samples, labels], x, discriminator_optimizer

def dim_ordering_fix(x):
    if K.image_dim_ordering() == 'th':
        return x
    else:
        return np.transpose(x, (0, 2, 3, 1))


def dim_ordering_unfix(x):
    if K.image_dim_ordering() == 'th':
        return x
    else:
        return np.transpose(x, (0, 3, 1, 2))


def dim_ordering_shape(input_shape):
    if K.image_dim_ordering() == 'th':
        return input_shape
    else:
        return (input_shape[1], input_shape[2], input_shape[0])


def dim_ordering_input(input_shape, name):
    if K.image_dim_ordering() == 'th':
        return Input(input_shape, name=name)
    else:
        return Input((input_shape[1], input_shape[2], input_shape[0]), name=name)


def dim_ordering_reshape(k, w, **kwargs):
    if K.image_dim_ordering() == 'th':
        return Reshape((k, w, w), **kwargs)
    else:
        return Reshape((w, w, k), **kwargs)
