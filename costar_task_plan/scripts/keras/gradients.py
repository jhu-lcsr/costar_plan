#!/usr/bin/env python

from keras.layers import Input, Dense
from keras.models import Model
import keras.backend as K

import tensorflow as tf
sess = tf.Session()
K.set_session(sess)

#input = tf.placeholder(tf.float32, shape=(None, 2)) #Input(shape=[2])
input = Input(shape=[2])
probs = Dense(1, activation='linear', init='normal')(input)

model = Model(input, probs)
model.summary()

sess.run(tf.initialize_all_variables())

with sess.as_default():
    print sess.run([probs], feed_dict={input: [[1,1]], K.learning_phase(): 0})
    print sess.run([probs], feed_dict={input: [[2,1]], K.learning_phase(): 0})
    print sess.run([probs], feed_dict={input: [[1,0]], K.learning_phase(): 0})

    grads = tf.gradients(probs, input)
    print grads

batch_size = 1
weights = model.trainable_weights # weight tensors


print "====================="
print "WEIGHTS:"

#weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
#gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
if K._BACKEND == 'tensorflow':
  gradients = K.gradients(model.output, model.trainable_weights)
  gradients = [g / float(batch_size) for g in gradients]  # since TF sums over the batch
elif K._BACKEND == 'theano':
  import theano.tensor as T
  gradients = T.jacobian(model.output.flatten(), model.trainable_weights)
  gradients = [K.mean(g, axis=0) for g in gradients]
else:
  raise RuntimeError('Unknown Keras backend "{}".'.format(K._BACKEND))

for (wt, g) in zip(weights, gradients):
    print "--- wts/grads:"
    print sess.run(wt)

print "-------------------------"
print "Layers", model.layers
print "-------------------------"
print "Weights", weights
print "-------------------------"
print "Gradients", gradients
# ==> [dense_1_W, dense_1_b]


import keras.backend as K

'''
input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]

get_gradients = K.function(inputs=input_tensors, outputs=gradients)

from keras.utils.np_utils import to_categorical

inputs0 = [[[1, 2]], # X
          [1], # sample weights
          [[1]], # y
          0 # learning phase in TEST mode
]
inputs1 = [[[1, 1]], # X
          [1], # sample weights
          [[1]], # y
          0 # learning phase in TEST mode
]
inputs2 = [[[1, 2]], # X
          [1], # sample weights
          [[0]], # y
          0 # learning phase in TEST mode
]

show_weights = [weight.value() for weight in weights]
#show_weights = [weight.initialized_value() for weight in weights]
print zip(show_weights, get_gradients(inputs0))
print zip(show_weights, get_gradients(inputs1))
print zip(show_weights, get_gradients(inputs2))
# ==> [(dense_1_W, array([[-0.42342907],
#                          [-0.84685814]], dtype=float32)),
#       (dense_1_b, array([-0.42342907], dtype=float32))]


print "================================"
print "================================"
print "================================"

'''

def collect_trainable_weights(layer):
    '''Collects all `trainable_weights` attributes,
    excluding any sublayers where `trainable` is set the `False`.
    '''
    trainable = getattr(layer, 'trainable', True)
    if not trainable:
        return []
    weights = []
    if layer.__class__.__name__ == 'Sequential':
        for sublayer in layer.flattened_layers:
            weights += collect_trainable_weights(sublayer)
    elif layer.__class__.__name__ == 'Model':
        for sublayer in layer.layers:
            weights += collect_trainable_weights(sublayer)
    elif layer.__class__.__name__ == 'Graph':
        for sublayer in layer._graph_nodes.values():
            weights += collect_trainable_weights(sublayer)
    else:
        weights += layer.trainable_weights
    # dedupe weights
    weights = list(set(weights))
    weights.sort(key=lambda x: x.name)
    return weights

# from https://github.com/fchollet/keras/issues/3062
import keras
def get_trainable_params(model):
    params = []
    for layer in model.layers:
        params += collect_trainable_weights(layer)
    return params

network_params = get_trainable_params(model)
#param_grad = tf.gradients(cost, network_params)

#import tensorflow as tf
# first define the model that takes state as input and outputs actions
#action = model(Input)

#x = tf.placeholder(tf.float32, shape=(None, 2))
#param_grad = tf.gradients(model.output, network_params)#, grad_ys=dQ_da)
#param_grad = tf.gradients(action, network_params, grad_ys=dQ_da)
# where dQ_da is grad of critic wrt action
#sess.run(param_grad, {x: [[1,0]]})

#print model.output
#print param_grad
#print grads([[1,2]])
