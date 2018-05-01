from __future__ import print_function

import h5py as h5f
import numpy as np
import os
import datetime

class H5fDataset(object):
    '''
    Write out h5f datasets one file at a time, with information in filenames
    for easy access and data aggregation.
    '''

    def __init__(self, name):
        '''
        Create a folder to hold different archives in
        '''
        self.name = os.path.expanduser(name)
        try:
            os.mkdir(self.name)
        except OSError as e:
            pass

    def write(self, example, filename, image_types=[]):
        '''
        Write an example out to disk.

        status: success, failure or error.failure
        '''
        filename = os.path.join(self.name, filename)
        f = h5f.File(filename, 'w')
        for key, value in example.items():
            print('writing key: ' + str(key))
            f.create_dataset(key, data=value)
        if image_types != []:
            #dt = h5f.special_dtype(vlen=bytes)
            for (img_type_str, img_format_str) in image_types:
                f.create_dataset("type_" + img_type_str, data=[img_format_str])
        f.close()

    def load(self,success_only=False):
        '''
        Read a whole set of data files in. This part could definitely be
        faster/more efficient.

        Parameters:
        -----------
        success_only: exclude examples without the "success" tag in their
                      filename when loading. Good when learning certain types
                      of models.

        Returns:
        --------
        data: dict of features and other information.
        '''
        raise RuntimeError('h5f does not yet support direct loading of data')

    def preprocess(self, train=0.6, val=0.2):
        '''
        TODO(cpaxton) 
        Create train/val/test splits
        '''
        raise RuntimeError('h5f does not yet support train/test splits')
