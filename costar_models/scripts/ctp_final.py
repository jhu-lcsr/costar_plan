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

from costar_models.planner import *
from costar_models.multi import *

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

        print(">>> GOAL_CLASSIFIER")
        image_discriminator = LoadGoalClassifierWeights(model,
                make_classifier_fn=MakeImageClassifier,
                img_shape=(64, 64, 3))
        image_discriminator.compile(loss="categorical_crossentropy",
                                metrics=["accuracy"],
                                optimizer=model.getOptimizer())

        show = False
        correct_g1 = 0
        correct_g2 = 0
        total = 0
        err1_sum = 0.
        err2_sum = 0.
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
                res1 = np.argmax(image_discriminator.predict([x0, xg]), axis=1)
                res2 = np.argmax(image_discriminator.predict([x0, xg2]), axis=1)
                if res1[0] == o1[i]:
                    correct_g1 += 1
                if res2[0] == o2[i]:
                    correct_g2 += 1
                err1 = np.mean(np.abs((xg[0] - I_target[i])))
                err2 = np.mean(np.abs((xg2[0] - I_target2[i])))
                err1_sum += err1
                err2_sum += err2
                total += 1
                mean1 = err1_sum / total
                mean2 = err2_sum / total
                print(correct_g1, "/", total, correct_g2, "/", total, "...", o1[i], o2[i], res1[0], res2[0], "errs =", err1, err2, "means =", mean1, mean2)

    else:
        raise RuntimeError('Must provide a model to load')

if __name__ == '__main__':
    args = ParseModelArgs()
    if args['profile']:
        import cProfile
        cProfile.run('main(args)')
    else:
        main(args)
