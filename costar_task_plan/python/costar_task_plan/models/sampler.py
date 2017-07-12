from tensorflow import TensorShape

'''
Sampler Neural Net
This neural net replaces the Z(mu, sigma) distribution from which we draw
trajectories. It's an LSTM-RNN neural network. Input is the state of the sample
position, output is the 
'''
class SamplerNetwork(object):
  
  def __init__(self, feature_layers=[32,32], timesteps=5):
    # create a number of input layers
    for layer_size in feature_layers:
      pass


import keras.backend as K
import numpy as np

class SamplerLoss(object):
    def __init__(self):
        assert K.backend() == u'tensorflow'

    def __call__(self, x1, x2):
        nsamples = x1.shape[0]
        dims = list(x1.shape)
        x1 = K.reshape(x1, TensorShape([1,] + list(x1.shape)))
        x2 = K.reshape(x2, TensorShape([1,] + list(x2.shape)))
        x2 = K.permute_dimensions(x2, (1,0,2))
        tile_shape1 = TensorShape([nsamples,1,1])
        tile_shape2 = TensorShape([1,nsamples,1])
        print x1, x2
        rep1 = K.tile(x1, tile_shape1)
        rep2 = K.tile(x2, tile_shape2)
        print rep1, rep2
        # sanity check
        #diff = K.concatenate([rep1,rep2])
        diff = rep1 - rep2
        return diff
        #return K.dot(rep1, rep2)

if __name__=='__main__':
    A = np.array([[0,1,0],[1,0.7,0.5]]).T
    B = np.array([[0.001, 0.002, 2.01],[0.001, 0.002, 2.01]]).T
    loss = SamplerLoss()
    x = K.variable(value=A)
    y = K.variable(value=B)
    print K.eval(loss(x,y))

    #z = K.dot(x,y)
    #res =  K.eval(z)
    #print res
    



