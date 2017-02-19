

from costar_task_search.abstract import *

'''
GMM policy:
  - fit to training data of some sort
  - execution takes the current set of features
'''
class GmmPolicy(AbstractPolicy):

  def __init__(self, dataset):
    # TODO(cpaxton): fit to dataset
    pass

  '''
  Compute features if actor id is nonzero; use world
  '''
  def __call__(self, world, state, actor):
    pass
