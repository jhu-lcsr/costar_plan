
'''
Empty (for now) Config class to store variables
'''
class Config(object):

    def __init__(self, filename):
        self.filename = filename

'''
Datasets
- download: find them online at a default location and save to disk somewhere
- load: read from the disk
'''
class Dataset(object):

  def __init__(self, name):
      self.name = name

  def download(self, *args, **kwargs):
    raise RuntimeError('downloading this dataset is not yet supported')

  def load(self, *args, **kwargs):
    raise RuntimeError('loading this dataset is not yet supported')
