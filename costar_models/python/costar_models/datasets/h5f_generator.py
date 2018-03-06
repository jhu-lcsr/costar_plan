from __future__ import print_function

import h5py as h5f
import numpy as np
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

    def _load(self, filename):
        '''
        Helper to load the file
        '''
        data = {}
        with h5f.File(filename, 'r') as f:
            if "image" in f:
                s = f['image_type'][0]
                load_jpeg = s.lower() == "jpeg"
            else:
                load_jpeg = False
            for k, v in f.items():
                if k == "image_type":
                    continue
                if load_jpeg and k in ["image", "goal_image"]:
                    data[k] = ConvertJpegListToNumpy(v)
                else:
                    data[k] = np.array(v)        
        return data

