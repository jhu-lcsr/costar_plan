import keras
import numpy as np

'''
Fast version of a Keras network.
This will read through all your layers under certain assumptions.

Inputs:
  - a keras model
  - a function to apply to the outputs of this model

Internal layers are assumed to be followed by a ReLu.
'''
class FastNetwork(object):

  def __init__(self, model, activation = np.tanh):
    self.layers = []
    self.relu = []
    self.activation = np.tanh
    for layer in model.layers:
      if isinstance(layer, keras.layers.core.Dense):
        [W, d] = layer.get_weights()
        z = np.zeros(d.shape)
        self.layers.append((W,d,z))
      elif isinstance(layer, keras.layers.core.Activation):
        print "Skipping activation layer. This better be a ReLu or the final layer."

  def predict(self, f):
    for (W,d,z) in self.layers[:-1]:
      f = np.maximum(np.dot(f,W) + d,z)
    W,d,_ = self.layers[-1]
    return self.activation(np.dot(f,W) + d)
