#!/usr/bin/env python

import unittest

import keras.losses as l
import keras.backend as K
import numpy as np

from costar_models import SamplerLoss

class SamplerLossTest(unittest.TestCase):
    def test1(self):
        A = np.array([[0,1,0],[1,0.7,0.5]]).T
        B = np.array([[0.001, 1.002, 0.01],[0.001, 0.002, 2.01]]).T

        print "==========="

        correct = np.zeros((3,3))
        
        print "A 1\t\t2\t\t3"
        for i in xrange(3):
            for j in xrange(3):
                correct[i,j] = np.sum((B[i] - A[j])**2)
        print correct

        print "==========="

        loss = SamplerLoss()
        x = K.variable(value=A)
        y = K.variable(value=B)
        x2 = K.variable(value=np.array([A]))

        from keras.layers import Lambda
        import tensorflow as tf
        print "---"
        print x2
        print x2.shape
        print B
        print K.eval(Lambda(lambda x: tf.gather_nd(tf.transpose(x),[[i] for i in range(2)]))(x2))
        print K.eval(Lambda(lambda x: tf.gather_nd(x,[[0,1]]))(x2))
        print "---"

        # Distances from A to B
        res = K.eval(loss._dists(x,y))
        print res
        self.assertTrue(np.all(np.abs(correct - res) < 1e-5))

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
    def test2(self):
        A = np.array([[0,0,1,0,0]])
        B = np.array([[2]])
        C = np.array([[3]])
        A = K.variable(value=A)
        B = K.variable(value=B)
        C = K.variable(value=C)
        cc = l.get("categorical_crossentropy")
        print K.eval(cc(A,B))
        print K.eval(cc(A,C))
        print K.eval(cc(A,A))
    
if __name__ == '__main__':
    import tensorflow as tf
    with tf.device('/cpu:0'):
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(config=config)
        K.set_session(sess)
    unittest.main()
