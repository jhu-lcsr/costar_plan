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
