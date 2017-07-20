#adapted from:
# https://github.com/yanpanlau/DDPG-Keras-Torcs/blob/master/CriticNetwork.py

import numpy as np
import math
import keras
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, convolutional=False):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        self.convolutional = convolutional
        
        #K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):

        if self.convolutional:
            # state input and convos---------------------------------------
            #S = (Flatten(input_shape=(1,) + (state_size))
            
            S = Input(shape=(state_size[0], state_size[1], 1))
            #x = BatchNormalization(axis=1)(S)   
            #100x100 -> 31x31
            x = Conv2D(32, (7,7), strides=(3,3), activation="relu")(S)
            #x = BatchNormalization(axis=1)(x)
            #31x31 -> 10x10
            x = Conv2D(32, (5,5), strides=(3,3), activation="relu")(x)
            #x = BatchNormalization(axis=1)(x)
            #10x10 -> 4x4
            x = Conv2D(32, (4,4), strides=(2,2), activation="relu")(x)
            #x = BatchNormalization(axis=1)(x)
            x = Flatten()(x)
            x = Dense(128, activation="relu")(x) 
            #x = BatchNormalization()(x)

        else:
            S = Input(shape=(state_size[0],))
            #x = BatchNormalization()(S)
            x = Dense(128, activation="relu")(S)
            x = Dense(128, activation="relu")(x)

        # action inputs ----------------------------------------------
        A = Input(shape=(action_dim,))
        a = Dense(128, activation="relu")(A)
        #a = BatchNormalization()(a)

        # Q-value computation -------------
        q = keras.layers.concatenate([x, a])
        q = Dense(256, activation="relu")(q)
        q = Dense(256, activation="relu")(q)
        #q = BatchNormalization()(q)
        init = keras.initializers.RandomUniform(minval=-0.0003, maxval=0.0003, seed=None)
        q = Dense(1, activation="linear", kernel_initializer=init, bias_initializer='zeros')(q)

        #V = Dense(action_dim,activation='linear')(h3)   
        model = Model(inputs=[S,A],outputs=q)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 