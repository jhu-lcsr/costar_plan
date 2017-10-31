#!/usr/bin/env python

from __future__ import print_function

# ----------------------------------------
# Before importing anything else -- make sure we load the right library to save
# images to disk for saving images.
#import matplotlib as mpl
#mpl.use("Agg")

from costar_models import *
from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.npy_generator import NpzGeneratorDataset

def RunDecoderVisualizer(args):
    '''
    This function 

    Parameters:
    -----------
    args: output of ParseModelArgs function, a dictionary of model parameters
          that determine how we should create and run the architecture we want
          to test.
    '''
    if 'cpu' in args and args['cpu']:
        import tensorflow as tf
        import keras.backend as K

        with tf.device('/cpu:0'):
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
            sess = tf.Session(config=config)
            K.set_session(sess)

    data_file_info = args['data_file'].split('.')
    data_type = data_file_info[-1]
    if data_type == "npz":
        root = ""
        for i, tok in enumerate(data_file_info[:-1]):
            if i < len(data_file_info)-1 and i > 0:
                root += '.'
            root += tok
        dataset = NpzGeneratorDataset(root)
        data = dataset.load(success_only = args['success_only'])
    else:
        raise NotImplementedError('data type not implemented: %s'%data_type)

    if 'model' in args and args['model'] is not None:
        model = args["model"]
        if not model == "predictor":
            raise RuntimeError('Unsupported model type "%s"'%model)
        model = RobotMultiDecoderVisualizer(taskdef=None, **args)
        model.load(world=None,**data)
        for i in range(dataset.numTest()):
            data, success = dataset.loadTest(i)
            if success or not model.success_only:
                model.generateImages(**data)
    else:
        raise RuntimeError('Must specify model type to visualize')

if __name__ == '__main__':
    args = ParseModelArgs()
    RunDecoderVisualizer(args)
