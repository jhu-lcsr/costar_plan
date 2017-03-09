#!/usr/bin/env python

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Merge, Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.utils import np_utils

# as the first layer in a model
model = Sequential()
#model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# now model.output_shape == (None, 10, 8)

# subsequent layers: no need for input_shape
#model.add(TimeDistributed(Dense(32)))
model.add(TimeDistributed(Convolution2D(32,4,4,activation='relu',border_mode='same'),input_shape=(5,8,8,1)))
model.add(TimeDistributed(MaxPooling2D((2, 2), border_mode='same')))
model.add(TimeDistributed(Dropout(0.2)))
model.add(TimeDistributed(Flatten()))

# now model.output_shape == (None, 10, 32)

#model.add(LSTM(32))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.summary()
