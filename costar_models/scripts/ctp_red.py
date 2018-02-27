#!/usr/bin/env python

from __future__ import print_function

import matplotlib as mpl
#mpl.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from costar_models import *
from costar_models.data_utils import *
from costar_models.plotting import *
from costar_models.planner import GetOrderedList, PrintTopQ

def visualizeHiddenMain(args):
    '''
    Tool for running model training without the rest of the simulation/planning/ROS
    code. This should be more or less independent and only rely on a couple
    external features.
    '''
    np.random.seed(0)
    ConfigureGPU(args)
    data, dataset = GetDataset(args)

    if 'model' in args and args['model'] is not None:
        model = MakeModel(taskdef=None, **args)
        model.validate = True
        model.load(world=None,**data)
 
        prev_option = model.null_option

        for i in range(100):
            #I = np.ones((1,64,64,3))
            #I[:,:,:,1] = np.zeros((64,64))
            #I[:,:,:,0] = np.zeros((64,64))
            I = np.random.random((1,64,64,3))
            plt.figure()
            #plt.subplot(1,5,1)
            #Show(I[0])
            h = model.encode(I)
            h = np.random.random((1,8,8,8))
            plt.subplot(1,4,1)
            Show(np.mean(h[0],axis=-1))
            Id = model.decode(h)
            plt.subplot(1,4,2)
            Show(Id[0])

            for j in range(200):
                #h = model.transform(h,h,np.array([36]))
                h = model.transform(h,h,np.array([np.random.randint(model.num_options)]))
            h2 = h
            plt.subplot(1,4,3)
            Show(np.mean(h2[0],axis=-1))

            Id = model.decode(h2)
            plt.subplot(1,4,4)
            Show(Id[0])

            plt.show()
             

    else:
        raise RuntimeError('Must provide a model to load')

if __name__ == '__main__':
    args = ParseModelArgs()
    if args['profile']:
        import cProfile
        cProfile.run('main(args)')
    else:
        visualizeHiddenMain(args)
