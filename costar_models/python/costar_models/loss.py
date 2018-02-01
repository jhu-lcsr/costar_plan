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

class KLDivergenceLoss(object):
    def __init__(self, mu, sigma):
        '''
        Set up the loss with mu and sigma -- the mean and covariance tensors.

        Parameters:
        -----------
        mu: mean tensor
        sigma: variance tensor. Technically not the covariance (as sigma would
               imply). And it's actually the log of the variance, so we can
               compute using gaussian noise.
        '''
        self.mu = mu
        self.sigma = sigma

    def __call__(self, target, pred):
        '''
        Compute the KL divergence portion of the loss term, so we can learn a
        mean and a variance for the underlying data.
        '''
        kl_loss = -0.5 * K.sum(1 + self.sigma - K.square(self.mu) -
                K.exp(self.sigma), axis=-1)
        return K.mean(kl_loss)

class EncoderDistance(object):
    def __init__(self, encoder, loss):
        self.encoder = encoder
        self.loss = keras.losses.get(loss)

    def __call__(self, target, pred):
        encoded_targets = self.encoder.predict(target)
	return self.loss(encoded_targets, pred)
