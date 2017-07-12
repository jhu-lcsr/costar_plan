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
    def __init__(self, sum_axis=-1, min_axis=1, eps=1e-8):
        '''
        Min axis should give us the nearest neighbor of the second parameter,
        since this is the set of predicted values.
        '''
        assert K.backend() == u'tensorflow'
        self.sum_axis = sum_axis
        self.min_axis = min_axis
        self.eps = eps

    def _dists(self, x1, x2):
        '''
        Compute a matrix of distances between every entry and every other
        entry. Assume all are vectors and sum along the given axis. We're
        really assuming this is the last axis here.
        '''
        nsamples = x1.shape[0]
        dims = list(x1.shape)
        x1 = K.reshape(x1, TensorShape([1,] + list(x1.shape)))
        x2 = K.reshape(x2, TensorShape([1,] + list(x2.shape)))
        x2 = K.permute_dimensions(x2, (1,0,2))
        tile_shape1 = TensorShape([nsamples,1,1])
        tile_shape2 = TensorShape([1,nsamples,1])
        #print x1, x2
        rep1 = K.tile(x1, tile_shape1)
        rep2 = K.tile(x2, tile_shape2)
        #print rep1, rep2
        # sanity check
        #diff = K.concatenate([rep1,rep2])
        x = rep1 - rep2
        x = K.square(x)
        x = K.sum(x, axis=self.sum_axis)
        return x

    def __call__(self, target, pred):
        '''
        Compute the distance to the nearest neighbor associated with each
        target, and return this value.
        '''
        d_tp = self._dists(target, pred)
        d_tp = K.min(d_tp,axis=self.min_axis,keepdims=True)

        # So right now this all comes out to just zeros...
        d_pp = self._dists(pred, pred)
        d_pp = K.min(d_pp,axis=self.min_axis,keepdims=True)
        d_pp = d_pp + (1e10 * K.eye(d_pp.shape[0]))

        # Add a dimension and some ones.
        nsamples = target.shape[0]
        ones = K.ones_like(d_pp) * self.eps
        d_pp = K.max(K.concatenate([ones, K.sqrt(d_pp)]),axis=1,keepdims=True)
        
        # This gives us one loss per sample. The sum is kind of unnecessary in
        # the current implementation.
        loss = -1. * K.sum(K.log(d_pp) / K.sqrt(d_tp), axis=1, keepdims=True)

        return loss

if __name__=='__main__':
    A = np.array([[0,1,0],[1,0.7,0.5]]).T
    B = np.array([[0.001, 1.002, 0.01],[0.001, 0.002, 2.01]]).T

    print "==========="

    correct = np.zeros((3,3))
    
    print "A 1\t\t2\t\t3"
    for i in xrange(3):
        for j in xrange(3):
            print "%f\t"%np.sum((B[i] - A[j])**2),
        print ""

    print "==========="

    loss = SamplerLoss()
    x = K.variable(value=A)
    y = K.variable(value=B)

    # Distances from A to B
    res = K.eval(loss._dists(x,y))
    print res

    # Distances from B to itself
    res = K.eval(loss._dists(y,y))
    print res

    print "==========="
    print "Loss"
    print "==========="

    # Loss
    res2 = K.eval(loss(x,y))
    print res2

    #z = K.dot(x,y)
    #res =  K.eval(z)
    #print res
    



