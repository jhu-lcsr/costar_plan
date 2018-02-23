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

        np.random.seed(0)
        features, targets = next(test_generator)
        [I0, I, o1, o2, oin] = features
        [ I_target, I_target2, o1_1h, value, qa, ga, o2_1h] = targets

        # Same as in training code
        model.model.predict([I0, I, o1, o2, oin])
        h = model.encode(I)
        h0 = model.encode(I0)
        prev_option = oin
        null_option = np.ones_like(prev_option) * model.null_option
        p_a, done1 = model.pnext(h0, h, prev_option)
        q_a, _ = model.q(h0, h, prev_option)
        q = model.q(h0, h, prev_option)
        v = model.value(h0, h)

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
        v_goal3 = model.value(h0, h_goal3)
        print("--------------\nHidden state:\n--------------\n")
        print("shape of hidden samples =", h.shape)
        print("shape of images =", I.shape)
        for i in range(h.shape[0]):
            print("------------- %d -------------"%i)
            print("prev option =", prev_option[i])
            print("best option =", action[i], action2[i], action3[i])
            print("done = ", done1[i], done2[i], done3[i])
            print("actual option=", o1[i], o2[i])
            print("value =", v[i], "actual =", value[i])
            print("goal value =", v_goal[i], v_goal2[i])
            pa_idx1 = GetOrderedList(p_a[i])
            qa_idx1 = GetOrderedList(q_a[i])
            pa_idx2 = GetOrderedList(p_a2[i])
            qa_idx2 = GetOrderedList(q_a2[i])
            pa_idx3 = GetOrderedList(p_a3[i])
            qa_idx3 = GetOrderedList(q_a3[i])
            print(" --- 1 ---")
            print(pa_idx1)
            print(qa_idx1)
            PrintTopQ(pa_idx1, q_a, i, n=4)
            print(" --- 2 ---")
            print(pa_idx2)
            print(qa_idx2)
            PrintTopQ(pa_idx2, q_a2, i, n=4)
            print(" --- 3 ---")
            print(pa_idx3)
            print(qa_idx3)
            PrintTopQ(pa_idx3, q_a3, i, n=4)

            plt.figure()
            plt.subplot(4,4,5)
            Show(I[i])
            plt.subplot(4,4,1)
            Show(I0[i])
            h_avg = np.squeeze(np.mean(h[i],axis=-1))
            h0_avg = np.squeeze(np.mean(h0[i],axis=-1))
            hg1_avg = np.squeeze(np.mean(h_goal[i],axis=-1))
            hg2_avg = np.squeeze(np.mean(h_goal2[i],axis=-1))
            hg3_avg = np.squeeze(np.mean(h_goal3[i],axis=-1))
            plt.subplot(4,4,2)
            Show(h0_avg)

            # Show the ground truth
            plt.subplot(4,4,3)
            Show(targets[0][i])
            plt.subplot(4,4,4)
            Show(targets[1][i])

            # Show the first image
            plt.subplot(4,4,6)
            Show(h_avg)
            plt.subplot(4,4,7)
            Show(hg1_avg)
            plt.subplot(4,4,8)
            Show(img_goal[i])

            # Show the second image
            plt.subplot(4,4,11)
            Show(hg2_avg)
            plt.subplot(4,4,12)
            Show(img_goal2[i])

            # Show the third image
            plt.subplot(4,4,15)
            Show(hg3_avg)
            plt.subplot(4,4,16)
            Show(img_goal3[i])

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
