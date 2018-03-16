#!/usr/bin/env python

from __future__ import print_function

from scipy.misc import imresize

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import h5py

from costar_models.datasets.image import GetJpeg, JpegToNumpy



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

    for fnum, filename in enumerate(dataset.test + dataset.train):
        pass


if __name__ == '__main__':
    args = ParseModelArgs()
    if args['profile']:
        import cProfile
        cProfile.run('main(args)')
    else:
        main(args)
