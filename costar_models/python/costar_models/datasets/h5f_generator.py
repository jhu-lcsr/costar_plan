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
        self.file = None

    # Interface for with statement
    def __enter__(self):
        return self.file

    def __exit__(self, *args):
        self.file.close()
        self.file = None

    def _load(self, filename, verbose=0):
        '''
        Helper to load the hdf5 file data into a dictionary of numpy arrays in memory.
        '''
        if verbose > 0:
            print('loading hf5 filename: ' + str(filename))
            debug_str = 'loading data:\n'

        dataset = h5f.File(filename, 'r')
        self.file = dataset

        # TODO: fix up this horrible code
        for k, v in six.iteritems(dataset):
            if verbose > 0:
                debug_str += 'key: ' + str(k)
            if k == "image_type" or k == "type_image":
                if verbose > 0:
                    debug_str += ' skipped\n'
                continue
            if verbose > 0:
                debug_str += ' added\n'
            if k == "image":
                self.load_jpeg.append(k)
            elif k == "depth_image":
                self.load_png.append(k)
        if verbose > 0:
            print(debug_str)
        return self
