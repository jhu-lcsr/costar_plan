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
    def __init__(self, name, split=0.1, ):
        '''
        Set name of directory to load files from

        '''
        self.name = name 
        self.split = split
        self.train = []
        self.test = []

    def _load(self, filename):
        '''
        Helper to load the file
        '''
        f = h5f.file(filename, 'r')
        return f

