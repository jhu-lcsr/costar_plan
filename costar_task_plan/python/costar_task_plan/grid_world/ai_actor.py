
"""
(c) 2016 Chris Paxton
"""

import tensorflow as tf
from actor import *


class RegressionActor(Actor):

    def __init__(self, state,
                 name="R", sample=False, miny=0):
        super(RegressionActor, self).__init__(state, name)
        self.model = None
        self.model_input = None
        self.sample = sample
        self.miny = miny

    '''
    policy learning:
    instead of argmax, randomly sample an action according to probabilities
    '''

    def sampleAction(self, world):
        if self.model is not None:
            nw = world.getLocalWorld(self)
            features = np.expand_dims(
                nw.getFeatures(self, useIntersection=True, flattened=True),
                axis=0)
            y = self.model.eval(
                feed_dict={self.model_input: features}) + self.miny
            y[y < 0] = 0
            yn = y / np.sum(y)  # normalize
            r = np.random.random(yn.shape[0])
            idx = np.argmin(yn.cumsum() < r)
            # print yn.cumsum()

            return self.actions[idx]
        else:
            return None

    '''
    how do we choose the next action
    '''

    def chooseAction(self, world):
        if self.sample:
            return self.sampleAction(world)
        else:
            return self.bestAction(world)

    '''
    choose highest probability actiona
    '''

    def bestAction(self, world):
        if self.model is not None:
            nw = world.getLocalWorld(self)
            features = np.expand_dims(
                nw.getFeatures(self, useIntersection=True, flattened=True),
                axis=0)
            y = self.action_index.eval(feed_dict={self.model_input: features})

            return self.actions[y[0]]
        else:
            return None

    def setModel(self, model, x, num_features):
        self.num_features = num_features
        self.model_input = x
        self.model = model
        self.action_index = tf.argmax(self.model, 1)

