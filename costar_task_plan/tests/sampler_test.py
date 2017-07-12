#!/usr/bin/env python

import unittest

import numpy as np

from costar_task_plan.models import SamplerLoss

class SamplerLossTest(unittest.TestCase):
    def test1(self):
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

        loss = FullSamplerLoss()
        x = K.variable(value=A)
        y = K.variable(value=B)

        # Distances from A to B
        res = K.eval(loss._dists(x,y))
        print res
        self.assertAlmostEqual(correct, res)

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
    



