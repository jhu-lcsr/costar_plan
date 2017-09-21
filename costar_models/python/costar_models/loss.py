from __future__ import print_function

import keras
import keras.backend as K

class BestHypothesisLoss(object):

    def __init__(self, num_hypotheses, num_actions, labels):
        self.num_hypotheses = num_hypotheses
        self.num_actions = num_actions
        # Labels should be num_hypotheses * num_actions
        self.labels = labels
        self.__name__ = "best_hypothesis"

    def __call__(self, target, pred):
        x = K.softmax(pred)
        x = K.expand_dims(pred,axis=-1)
        x = K.repeat_elements(x, self.num_actions, axis=-1)
        # multiply each one by its probability
        x = x * self.labels
        x = K.sum(x,axis=1)
        return keras.losses.binary_crossentropy(target, pred)
