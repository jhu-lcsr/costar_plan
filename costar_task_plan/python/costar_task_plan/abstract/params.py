
import numpy as np

'''
Really simple class meant to store world configuration params
'''
class Params(object):
  def __init__(self, shape):
    self.data = np.random.random(shape)
    self.idx = 0

  def length(self):
    return self.data.shape[0]

  def next(self):
    param = self.data[self.idx]
    self.idx += 1
    return param

  def nextint(self, min, max):
    param = self.next()
    return int(param*(max-min)) + min

  def get(self, idx):
    return self.data[idx]

  def reset(self):
    self.idx = 0
