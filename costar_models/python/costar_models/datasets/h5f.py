from __future__ import print_function

import h5py as h5f
import numpy as np
import os
import datetime

# TODO(ahundt) move to a utilities location
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Apply a timestamp to the front of a filename description.

    see: http://stackoverflow.com/a/5215012/99379
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

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

    def write(self, example, i, r, image_type=None):
        '''
        Write an example out to disk.
        '''
        if r > 0.:
            status = "success"
        else:
            status = "failure"
        filename = timeStamped("example%06d.%s.h5f"%(i,status))
        filename = os.path.join(self.name, filename)
        f = h5f.File(filename, 'w')
        for key, value in example.items():
            f.create_dataset(key, data=value)
        if image_type is not None:
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset("image_type", data=["image_type"])
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
