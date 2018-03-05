#!/usr/bin/env python

from __future__ import print_function

import matplotlib as mpl
#mpl.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import imageio

from costar_models import *
from costar_models.planner import GetOrderedList, PrintTopQ
from costar_models.sampler2 import PredictionSampler2
from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.npy_generator import NpzGeneratorDataset
from costar_models.datasets.h5f_generator import H5fGeneratorDataset

from costar_models.planner import *
from costar_models.multi import *
from costar_models.dvrk import MakeJigsawsImageClassifier

def main(args):
    '''
    Tool for running model training without the rest of the simulation/planning/ROS
    code. This should be more or less independent and only rely on a couple
    external features.
    '''
    ConfigureGPU(args)

    np.random.seed(0)
    data_file_info = args['data_file'].split('.')
    data_type = data_file_info[-1]
    root = ""
    for i, tok in enumerate(data_file_info[:-1]):
        if i < len(data_file_info)-1 and i > 0:
            root += '.'
        root += tok
    if data_type == "npz":
        dataset = NpzGeneratorDataset(root)
        data = dataset.load(success_only = args['success_only'])
    elif data_type == "h5f":
        dataset = H5fGeneratorDataset(root)
        data = dataset.load(success_only = args['success_only'])
    else:
        raise NotImplementedError('data type not implemented: %s'%data_type)

    if 'features' not in args or args['features'] is None:
        raise RuntimeError('Must provide features specification')
    features_arg = args['features']

    if 'model' not in args or args['model'] is None:
        raise RuntimeError('Must provide a model to load')
    model_arg = args['model']

    model = MakeModel(taskdef=None, **args)
    model.validate = True
    model.load(world=None,**data)
    train_generator = model.trainGenerator(dataset)
    test_generator = model.testGenerator(dataset)

    show = False
    correct_g1 = 0
    correct_g2 = 0
    total = 0.
    err1_sum = 0.
    err2_sum = 0.
    v_sum = 0.
    ii = 0
    imgs = []
    ht = 96
    w = 128
    for filename in dataset.test:
        print(filename)
        data = dataset.loadFile(filename)
        features, targets = model._getData(**data)
        [I0, I, o1, o2, oin] = features
        length = I0.shape[0]
        [I_target, I_target2] = targets[:2]
        img = np.zeros((200,260,3))
        for i in range(length):
            ii += 1
            xi = np.expand_dims(I[i],axis=0)
            x0 = np.expand_dims(I0[i],axis=0)
            prev_option = np.array([oin[i]])
            h = model.encode(xi)
            h0 = model.encode(x0)
            #p_a, done1 = model.pnext(h0, h, prev_option)
            #v2 = model.value(h0, h)
            h_goal = model.transform(h0, h, np.array([o1[i]]))
            h_goal2 = model.transform(h0, h_goal, np.array([o2[i]]))
            xg = model.decode(h_goal)
            xg2 = model.decode(h_goal2)
            if show:
                plt.subplot(1,4,1); plt.imshow(x0[0])
                plt.subplot(1,4,2); plt.imshow(xi[0])
                plt.subplot(1,4,3); plt.imshow(xg[0])
                plt.subplot(1,4,4); plt.imshow(xg2[0])
                plt.show()
            err1 = np.mean(np.abs((xg[0] - I_target[i])))
            err2 = np.mean(np.abs((xg2[0] - I_target2[i])))
            yimg, ximg = 70, 3
            img[yimg:(yimg+ht),ximg:(ximg+w),:] = xi[0]
            yimg, ximg = 5, 130
            img[yimg:(yimg+ht),ximg:(ximg+w),:] = xg[0]
            yimg, ximg = 100, 130
            img[yimg:(yimg+ht),ximg:(ximg+w),:] = xg2[0]
            #plt.imshow(img)
            #plt.show()
            imgs.append(img)
            #v = model.value(h0, h_goal2)
            #if v[0] > 0.5 and value[i] > 0.5:
            #    vacc = 1.
            #elif v[0] < 0.5 and value[i] < 0.5:
            #    vacc = 1.
            #else:
            #    vacc = 0.
            err1_sum += err1
            err2_sum += err2
            total += 1.
            #v_sum += vacc
            mean1 = err1_sum / total
            mean2 = err2_sum / total
            print( o1[i], o2[i],
                    "means =", mean1, mean2,
                    "avg =", v_sum/total )
        break
    imageio.mimsave('movie.gif', imgs)


if __name__ == '__main__':
    args = ParseModelArgs()
    if args['profile']:
        import cProfile
        cProfile.run('main(args)')
    else:
        main(args)
