from __future__ import print_function

from keras import layers
import keras
import keras.backend as K
import tensorflow as tf

def mhp_loss_layer(num_classes, num_hypotheses, y_true, y_pred):
    '''
    This is the original code from Christian, for reference.
    '''
    xsum = tf.zeros([1, 1])
    xmin = tf.ones([1, 1])*1e10
    for i in range(0, num_hypotheses):
        cc = losses.categorical_crossentropy(y_true, tf.slice(y_pred, [0,
            num_classes*i], [1, num_classes]))
        xsum += cc
        xmin = tf.minimum(xmin, cc)

    return 0.05 * xsum / num_hypotheses + 0.90 * xmin

class MhpLoss(object):
    '''
    Defines Christian Rupprecht's multiple-hypothesis loss function. This one
    operates on multiple hypothesis samples. This version is designed for use
    with data of one type of output (e.g. an image).

    ArXiv: https://arxiv.org/pdf/1612.00197.pdf

    BibTex:
    @article{rupprecht2016learning,
      title={Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses},
      author={Rupprecht, Christian and Laina, Iro and Baust, Maximilian and Tombari, Federico and Hager, Gregory D and Navab, Nassir},
      journal={arXiv preprint arXiv:1612.00197},
      year={2016}
    }
    '''

    def __init__(self, num_hypotheses, avg_weight=0.05, loss="mae"):
        '''
        Set up the MHP loss function.

        Parameters:
        -----------
        num_hypotheses: number of targets to generate (e.g., predict 8 possible
                        future images).
        num_outputs: number of output variables per hypothesis (e.g., 64x64x3
                     for a 64x64 RGB image). Currently deprecated, but may be
                     necessary later on.
        '''
        self.num_hypotheses = num_hypotheses
        self.__name__ = "mhp_loss"
        
        if avg_weight > 0.25 or avg_weight < 0.:
            raise RuntimeError('avg_weight must be in [0,0.25]')
        self.avg_weight = avg_weight
        #self.min_weight = 1.0 - (2 * self.avg_weight)
        self.min_weight = 1.0 - self.avg_weight
        self.kl_weight = 0.001
        self.loss = keras.losses.get(loss)

    def __call__(self, target, pred):
        '''
        Pred must be of size:
            [batch_size=None, num_samples, traj_length, feature_size]
        Targets must be of size:
            [batch_size=None, traj_length, feature_size]

        You can use the tools in "split" to generate this sort of data (for
        targets). The actual loss function is just the L2 norm between each
        point.
        '''

        # Create tensors to hold outputs
        xsum = tf.zeros([1, 1])
        # Small value to make sure min is never 0 or negative
        xmin = tf.ones([1, 1])*1e10
    
        # Loop and apply MSE for all images
        for i in range(self.num_hypotheses):
            target_image = target[:,0]
            pred_image = pred[:,i]
            #cc = losses.mean_squared_error(target_image, pred_image)
            cc = self.loss(target_image, pred_image)
            xsum += cc
            xmin = tf.minimum(xmin, cc)

        return (self.avg_weight * xsum / self.num_hypotheses) \
                + (self.min_weight * xmin)

class MhpLossWithShape(object):
    '''
    This version of the MHP loss assumes that it will receive multiple outputs.

    '''
    def __init__(self, num_hypotheses, outputs, weights=None, loss="mse",
            avg_weight=0.05, stats=[]):
        '''

        Parameters:
        -----------
        num_hypotheses: number of hypotheses 
        outputs: length of each output 
        weights: None or vector of weights for each target
        loss: loss function or vector of loss function names to use (keras)
        avg_weight: amount of weight to give to average loss across all
                    hypotheses
        stats: mean, log variance of Gaussian from which each hypothesis was
               drawn, used to add a KL regularization term to the weight
        '''
        self.kl_weight = 1e-8
        self.num_hypotheses = num_hypotheses
        self.outputs = outputs # these are the sizes of the various outputs
        if weights is None:
            self.weights = [1.] * len(self.outputs)
        else:
            self.weights = weights
        if stats is not None and len(stats) > 0:
            if len(stats) == 1:
                stats = stats * self.num_hypotheses
                self.stats = stats
            elif len(stats) == self.num_hypotheses:
                self.stats = stats
            else:
                raise RuntimeError('statistics vector not acceptable length')
        else:
            self.stats = None
        assert len(self.weights) == len(self.outputs)
        self.losses = []
        if isinstance(loss, list):
            for loss_name in loss:
                self.losses.append(keras.losses.get(loss_name))
        else:
            self.losses = [loss] * len(self.outputs)
        assert len(self.outputs) == len(self.losses)
        self.__name__ = "mhp_loss"

        if avg_weight > 1.0 or avg_weight < 0.:
            raise RuntimeError('avg_weight must be in [0,1]')
        self.avg_weight = avg_weight
        #self.min_weight = 1.0 - (2 * self.avg_weight)
        self.min_weight = 1.0 - self.avg_weight

    def __call__(self, target, pred):
        '''
        Pred must be of size:
            [batch_size=None, num_samples, traj_length, feature_size]
        Targets must be of size:
            [batch_size=None, traj_length, feature_size]

        Where:
            feature_size = product(self.outputs)
        i.e., the feature size is the sum of the sizes of all outputs. All
        outputs must be pulled in order.
        '''

        xsum = tf.zeros([1, 1])
        xmin = tf.ones([1, 1])*1e10


        for i in range(self.num_hypotheses):

            target_outputs = _getOutputs(target, self.outputs, 0)
            pred_outputs = _getOutputs(pred, self.outputs, i)
            
            # Hold loss for all outputs here.
            cc = tf.zeros([1,1])
            for wt, target_out, pred_out, loss in zip(self.weights, target_outputs,
                    pred_outputs, self.losses):
                # loss = feature weight * MSE for this feature
                loss_term = loss(target_out, pred_out)
                while len(loss_term.shape) > 1:
                    # remove axes one at a time
                    loss_term = K.mean(loss_term,axis=-1)
                cc += wt * loss_term

            cc = cc / len(self.outputs)
            if self.stats is not None:
                mu, sigma = self.stats[i]
                kl_loss = -0.5 * K.sum(1 + sigma - K.square(mu) -
                        K.exp(sigma), axis=-1)
                cc += self.kl_weight * kl_loss

            xsum += cc
            xmin = tf.minimum(xmin, cc)

        return ((self.avg_weight * xsum / self.num_hypotheses)
            + (self.min_weight * xmin))

def _getOutputs(state, outputs, i):
    '''
    Split a single output vector into multiple targets. This is a work-around
    because you can't have a Keras loss function that operates over multiple
    outputs.

    Parameters:
    -----------
    state: vector of data to split
    ouputs: dimensionality of each output to retrieve in order
    '''
    idx = 0
    separated_outputs = []
    for output_dim in outputs:
        # Print statement for debugging: shows ranges for each output, which
        # should match the order of provided data.
        #print("from ", idx, "to", idx+output_dim)
        out = state[:,i,idx:idx+output_dim]
        separated_outputs.append(out)
        idx += output_dim
    return separated_outputs
