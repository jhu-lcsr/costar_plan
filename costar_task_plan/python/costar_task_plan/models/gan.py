
import keras.backend as K
import numpy as np

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, RepeatVector, Reshape
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Dense, Conv2D, Activation
from keras.layers.merge import Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam

class GAN(object):
    def __init__(self, generator_vars, discriminator_vars, noise_dim):

        generator_inputs, self.generator, generator_optimizer = generator_vars
        discriminator_inputs, self.discriminator, discriminator_optimizer = discriminator_vars

        # =====================================================================
        # Set up adversarial model
        self.discriminator.trainable = False
        adversarial_graph = self.discriminator([self.generator.outputs[0], discriminator_inputs[1]])
        self.adversarial = Model(generator_inputs +
                [discriminator_inputs[1]], adversarial_graph)
        self.adversarial.compile(optimizer=generator_optimizer,
                                 loss="binary_crossentropy")

        self.noise_dim = noise_dim

        self.printSummary()

    def printSummary(self):
        #print self.generator.summary()
        #print self.discriminator.summary()
        print self.adversarial.summary()

    def fit(self, x, y, num_iter=1000, batch_size=50, save_interval=0):
        for i in xrange(num_iter):

            # Sample one batch, including random noise
            idx = np.random.randint(0, y.shape[0], size=batch_size)
            xi = x[idx]
            yi = y[idx]
            noise = np.random.random((batch_size, self.noise_dim))

            self.discriminator.trainable = True
            self.discriminator.trainable = False

            #loss = self.generator.train_on_batch([noise, yi], xi)
            loss = self.adversarial.train_on_batch(
                    [noise, yi, yi],
                    np.zeros((batch_size,)))
            print "Iter %d: loss = "%(i), loss

    def _adversarial(self):
        pass

    def _discriminator(self):
        pass

class SimpleGAN(GAN):
    '''
    Feed forward network -- good for simple data sets.
    '''
    
    def __init__(self, input_shape, output_shape):
        pass

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
        self.generator_filters_c1 = 128

        self.discriminator_dense_size = 1024
        self.discriminator_filters_c1 = 64

        generator_vars = self._generator(self.img_shape, labels, noise_dim)
        discriminator_vars = self._discriminator(self.img_shape, labels)

        super(SimpleImageGan, self).__init__(
                generator_vars,
                discriminator_vars,
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
        x = Dense(self.generator_dense_size)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Concatenate(axis=1)([x, labels])

        # Add second dense layer
        cnn_inputs = self.generator_filters_c1
        x = Dense(cnn_inputs * height4 * width4)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)

        # Add labels and adjust size -- append to every space
        labels2 = RepeatVector(height4*width4)(labels)
        labels2 = Reshape((height4,width4,num_labels))(labels2)
        x = Reshape((height4, width4, cnn_inputs))(x)
        x = Concatenate(axis=3)([x,labels2])

        # Apply the first convolution
        x = Conv2DTranspose(cnn_inputs,
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   padding="same")(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)

        # Add labels in again -- to each column
        labels2 = RepeatVector(height2*width2)(labels)
        labels2 = Reshape((height2,width2,num_labels))(labels2)
        x = Concatenate(axis=3)([x,labels2])

        # Apply the second convolution
        x = Conv2DTranspose(self.img_shape[2],
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   padding="same")(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('sigmoid')(x)

        generator_optimizer = Adam(lr=2e-4, beta_1=0.5)
        x = Model([noise, labels], x)
        x.compile(optimizer=generator_optimizer, loss="binary_crossentropy", \
            metrics=['accuracy'])

        return [noise, labels], x, generator_optimizer

    def _discriminator(self, img_shape, num_labels):
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
        labels = Input(shape=(num_labels,))
        labels2 = RepeatVector(height*width)(labels)
        labels2 = Reshape((height,width,num_labels))(labels2)

        # Add initial discriminator layer
        x = Concatenate(axis=3)([samples, labels2])
        x = Conv2D(channels + num_labels,
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Add conv layer with more filters
        labels2 = RepeatVector(height2*width2)(labels)
        labels2 = Reshape((height2,width2,num_labels))(labels2)
        x = Concatenate(axis=3)([x, labels2])
        x = Conv2D(self.discriminator_filters_c1 + num_labels,
                   kernel_size=[5, 5], 
                   strides=(2, 2),
                   padding="same")(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Add dense layer
        x = Reshape((-1,))(x)
        x = Concatenate(axis=1)([x, labels])
        x = Dense(self.discriminator_dense_size)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Single output -- sigmoid activation function
        x = Concatenate(axis=1)([x, labels])
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        discriminator_optimizer = Adam(lr=1e-4, beta_1=0.5)
        x = Model([samples, labels], x)
        x.compile(optimizer=discriminator_optimizer,
                  loss="binary_crossentropy", \
                  metrics=['accuracy'])

        return [samples, labels], x, discriminator_optimizer
