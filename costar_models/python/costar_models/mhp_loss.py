from keras import layers
from keras import losses
import tensorflow as tf

def mhp_loss_layer(num_classes, num_hypotheses, y_true, y_pred):
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
    operates on multiple hypothesis samples.
    '''

    def __init__(self, num_hypotheses, num_outputs):
        self.num_hypotheses = num_hypotheses
        self.num_outputs = num_outputs
        #for dim in output_shape:
        #    self.num_outputs *= dim
        #self.output_shape = output_shape
        self.__name__ = "mhp_loss"

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

        xsum = tf.zeros([1, 1])
        xmin = tf.ones([1, 1])*1e10
        print target.shape, pred.shape
        for i in range(0, self.num_hypotheses):
            #cc = losses.mean_squared_error(y_true,
            #        (y_true, tf.slice(y_pred, [0, num_classes*i], [1,
            #            num_classes])))
            cc = losses.mean_squared_error(target, pred[:,i,:,:,:])
            xsum += cc
            xmin = tf.minimum(xmin, cc)

        return 0.05 * xsum / self.num_hypotheses + 0.90 * xmin


