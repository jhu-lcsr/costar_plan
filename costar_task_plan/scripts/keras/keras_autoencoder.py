#!/usr/bin/env python

'''
Learning about Keras and autoencoders
From this tutorial: https://blog.keras.io/building-autoencoders-in-keras.html
'''

import keras
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, UpSampling3D
from keras.models import Model

input_img = Input(shape=(1, 28, 28))

x = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((1, 2))(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(28, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

'''
IMPORT DATA AND CONFIGURE TRAINING
'''
print "Importing dataset..."
from keras.datasets import mnist
print "done."

import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

'''
TRAIN ON DATA
'''

from keras.backend import backend
if not backend() == u'theano':
    from keras.callbacks import TensorBoard

    autoencoder.fit(x_train, x_train,
                    nb_epoch=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
else:
    autoencoder.fit(x_train, x_train,
                    nb_epoch=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test))



'''
VISUALIZE RESULTS
'''

decoded_imgs = autoencoder.predict(x_test)

try:
    import matplotlib.pyplot as plt

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
except ImportError, e:
    print e
