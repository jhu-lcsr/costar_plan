from __future__ import print_function

import h5py as h5f
import numpy as np
import six
import os

from .npy_generator import NpzGeneratorDataset
from .image import *

class H5fGeneratorDataset(NpzGeneratorDataset):
    '''
    Placeholder class for use with the ctp training tool. This one generates
    samples from an h5f encoded set of data.

    The Npz generator version implements a generic version of this that just
    takes the load function so all we need to do is implement things so they'll
    load a particular class.
    '''
    def __init__(self, *args, **kwargs):
        super(H5fGeneratorDataset, self).__init__(*args, **kwargs)
        self.file_extension = 'h5'

    def _load(self, filename, verbose=0):
        '''
        Helper to load the hdf5 file data into a dictionary of numpy arrays in memory.
        '''
        data = {}
        if verbose > 0:
            print('loading hf5 filename: ' + str(filename))
            debug_str = 'loading data:\n'
        with h5f.File(filename, 'r') as dataset:
            if "image" in dataset and "image_type" in dataset:
                s = dataset['image_type'][0]
                load_jpeg = s.lower() == "jpeg"
            elif "image" in dataset and "type_image" in dataset:
                s = dataset['type_image'][0]
                load_jpeg = s.lower() == "jpeg"
            else:
                load_jpeg = False
            for k, v in six.iteritems(dataset):
                if verbose > 0:
                    debug_str += 'key: ' + str(k)
                if k == "image_type" or k == "type_image":
                    if verbose > 0:
                        debug_str += ' skipped\n'
                    continue
                data[k] = np.array(v)
                # for debugging missing data, remove if it has been working:
                # if k in ['goal_idx', 'label']:
                #     print(str(k) + ': ' + str(data[k]))
                if verbose > 0:
                    debug_str += ' added\n'
                if k == "image" and len(data[k].shape) < 3:
                    load_jpeg = ["image"]
                if k == "depth_image" and len(data[k].shape) < 3:
                    load_png = ["depth_image"]
            self.load_jpeg += [load_jpeg]
            if verbose > 0:
                print(debug_str)
        return data
