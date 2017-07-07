
import keras.backend as K

from keras.layers import Input, RepeatVector, Reshape
from keras.layers import BatchNormalization
from keras.layers import Dense, Conv2D, Activation
from keras.layers.merge import Concatenate
from keras.models import Model

class GAN(object):
    def __init__(self, inputs, generator, discriminator, batch_size):
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size

        self.adversarial_model = discriminator(generator)

    def summary(self):
        print generator.summary()
        print discriminator.summary()

    def fit(self, x, y, num_iter=1000, save_interval=0):
        #idx = np.random.randint(0,
        #    self.x_train.shape[0], size=self.batch_size)
        #x = self.x_train[idx]
        #y = self.y_train[idx]
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
            noise_dim=100, batch_size=100,):
        img_shape = (img_rows, img_cols, channels)

        self.generator_dense_size = 1024
        self.generator_filters_c1 = 64

        self.discriminator_dense_size = 1024
        self.discriminator_filters_c1 = 64

        self.batch_size = batch_size

        generator = self._generator(img_shape, labels, noise_dim)
        generator.compile(optimizer="adam",loss="binary_crossentropy")
        print generator.summary()

    def _generator(self, img_shape, num_labels, noise_dim):
        height4 = img_shape[0]/4
        width4 = img_shape[0]/4

        labels = Input(shape=(num_labels,))
        noise = Input(shape=(noise_dim,))

        x = Concatenate(axis=1)([noise, labels])
        
        # Add first dense layer
        x = Dense(self.generator_dense_size)(x)
        x = BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Concatenate(axis=1)([x, labels])

        # Add second dense layer
        cnn_inputs = self.generator_filters_c1 * 2
        x = Dense(cnn_inputs * height4 * width4)(x)
        x = BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
        x = Activation('relu')(x)

        # add labels and adjust size -- append to every space
        labels2 = RepeatVector(height4*width4)(labels)
        labels2 = Reshape((height4,width4,num_labels))(labels2)
        x = Reshape((height4, width4, cnn_inputs))(x)
        x = Concatenate(axis=3)([x,labels2])
        
        return Model([noise, labels], x)
