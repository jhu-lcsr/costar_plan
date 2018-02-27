#!/usr/bin/env python

from __future__ import print_function

import matplotlib as mpl
#mpl.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from costar_models import *
from costar_models.plotting import *
from costar_models.planner import GetOrderedList, PrintTopQ
from costar_models.sampler2 import PredictionSampler2
from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.npy_generator import NpzGeneratorDataset
from costar_models.datasets.h5f_generator import H5fGeneratorDataset


def visualizeHiddenMain(args):
    '''
    Tool for running model training without the rest of the simulation/planning/ROS
    code. This should be more or less independent and only rely on a couple
    external features.
    '''
    ConfigureGPU(args)

    data_file_info = args['data_file'].split('.')
    data_type = data_file_info[-1]
    root = ""
    for i, tok in enumerate(data_file_info[:-1]):
        if i < len(data_file_info)-1 and i > 0:
            root += '.'
        root += tok

    np.random.seed(0)
    if data_type == "npz":
        dataset = NpzGeneratorDataset(root)
        data = dataset.load(success_only = args['success_only'])
    elif data_type == "h5f":
        dataset = H5fGeneratorDataset(root)
        data = dataset.load(success_only = args['success_only'])
    else:
        raise NotImplementedError('data type not implemented: %s'%data_type)

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
            h2 = model.transform(h,h,np.array([36]))
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
