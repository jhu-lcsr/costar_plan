from __future__ import print_function

import h5py as h5f
import numpy as np
import os

from .npy_generator import NpzGeneratorDataset

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

    def _load(self, filename):
        '''
        Helper to load the file
        '''
        f = h5f.File(filename, 'r')
        return f

