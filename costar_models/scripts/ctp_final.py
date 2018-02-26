#!/usr/bin/env python

from __future__ import print_function

import matplotlib as mpl
#mpl.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from costar_models import *
from costar_models.planner import GetOrderedList, PrintTopQ
from costar_models.sampler2 import PredictionSampler2
from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.npy_generator import NpzGeneratorDataset
from costar_models.datasets.h5f_generator import H5fGeneratorDataset


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

    if 'model' in args and args['model'] is not None:
        model = MakeModel(taskdef=None, **args)
        model.validate = True
        model.load(world=None,**data)
        train_generator = model.trainGenerator(dataset)
        test_generator = model.testGenerator(dataset)

        for filename in dataset.test:
            print(filename)
            data = dataset.loadFile(filename)
            length = data['example'].shape[0]
            features, targets = model._getData(**data)
            [I0, I, o1, o2, oin] = features
            [I_target, I_target2, o1_1h, value, qa, ga, o2_1h] = targets
            for i in range(length):
                xi = np.expand_dims(I[i],axis=0)
                x0 = np.expand_dims(I0[i],axis=0)
                prev_option = np.array([oin[i]])
                h = model.encode(xi)
                h0 = model.encode(x0)
                p_a, done1 = model.pnext(h0, h, prev_option)
                v2 = model.value(h0, h)
                h_goal = model.transform(h0, h, np.array([o1[i]]))
                h_goal2 = model.transform(h0, h_goal, np.array([o2[i]]))
                xg = model.decode(h_goal)
                xg2 = model.decode(h_goal2)
                print(p_a)
                print(np.argmax(p_a, axis=1), o1[i], oin[i])
                plt.subplot(1,4,1); plt.imshow(x0[0])
                plt.subplot(1,4,2); plt.imshow(xi[0])
                plt.subplot(1,4,3); plt.imshow(xg[0])
                plt.subplot(1,4,4); plt.imshow(xg2[0])
                res1 = model.discriminator(xg)
                res2 = model.discriminator(xg2)
                print(o1, o2, res1, res2)
                plt.show()

        # Same as in training code
        #q_a, _ = model.q(h0, h, prev_option)
        #q = model.q(h0, h, prev_option)

        if not h.shape[0] == I.shape[0]:
            raise RuntimeError('something went wrong with dimensions')
        print("shape =", p_a.shape)
        action = np.argmax(p_a,axis=1)
        # Compute effects of first action
        #h_goal = model.transform(h0, h, o1)
        h_goal = model.transform(h0, h, action)
        p_a2, done2 = model.pnext(h0, h_goal, action)
        q_a2, _ = model.q(h0, h_goal, action)
        action2 = np.argmax(p_a2,axis=1)

        # Comute effects of next action
        #h_goal2 = model.transform(h0, h_goal, o2)
        h_goal2 = model.transform(h0, h_goal, action2)
        p_a3, done3 = model.pnext(h0, h_goal2, action2)
        q_a3, _ = model.q(h0, h, action2)
        action3 = np.argmax(p_a3,axis=1)

        # Comute effects of next action
        h_goal3 = model.transform(h0, h_goal2, action3)
        p_a4, done4 = model.pnext(h0, h_goal3, action3)
        q_a4, _ = model.q(h0, h,action3)
        action4 = np.argmax(p_a4,axis=1)

        # Compute values and images
        img_goal = model.decode(h_goal)
        img_goal2 = model.decode(h_goal2)
        img_goal3 = model.decode(h_goal3)
        v_goal = model.value(h0, h_goal)
        v_goal2 = model.value(h0, h_goal2)

    else:
        raise RuntimeError('Must provide a model to load')

if __name__ == '__main__':
    args = ParseModelArgs()
    if args['profile']:
        import cProfile
        cProfile.run('main(args)')
    else:
        main(args)
