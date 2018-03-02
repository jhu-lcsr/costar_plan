#!/usr/bin/env python

from __future__ import print_function

import matplotlib as mpl
#mpl.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from costar_models import *
from costar_models.data_utils import *
from costar_models.planner import GetOrderedList, PrintTopQ

def visualizeHiddenMain(args):
    '''
    Compute multiple parallel predictions from the first frame and display
    them. This script will only predict from h0, not from subsequent frames!
    '''
    np.random.seed(0)
    ConfigureGPU(args)
    data, dataset = GetDataset(args)

    if 'model' in args and args['model'] is not None:
        model = MakeModel(taskdef=None, **args)
        model.validate = True
        model.load(world=None,**data)
        train_generator = model.trainGenerator(dataset)
        test_generator = model.testGenerator(dataset)

        features, targets = next(test_generator)
        [I0, I, o1, o2, oin] = features
        if model.features == "multi" or model.features == None:
            [ I_target, I_target2, o1_1h, value, qa, ga, o2_1h] = targets
        else:
            [ I_target, I_target2, o1_1h, value, a, o2_1h] = targets

        # Same as in training code
        model.model.predict([I0, I, o1, o2, oin])
        h = model.encode(I)
        h0 = model.encode(I0)
        prev_option = oin
        null_option = np.ones_like(prev_option) * model.null_option
        p_a, _ = model.pnext(h0, h0, null_option)
        v = model.value(h0, h, prev_option)

        if not h.shape[0] == I.shape[0]:
            raise RuntimeError('something went wrong with dimensions')
        for i in range(h.shape[0]):
            print(p_a[i])
            pa_idx = GetOrderedList(p_a[i])
            n = 4
            plt.figure()
            plt.subplot(1,n+1,1)
            Show(I0[i])
            for j in range(n):
                action = np.array([pa_idx[j]])
                H0 = np.array([h0[i]])
                h = model.transform(H0, H0, action)
                _, done = model.pnext(H0, h, action)
                print("option =", j, "val =", model.value(H0, h, prev_option), "is done =", done)
                I = model.decode(h)
                plt.subplot(1,n+1,j+2)
                Show(I[0])

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
