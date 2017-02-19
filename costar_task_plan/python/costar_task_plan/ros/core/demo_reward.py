from costar_task_search.abstract import AbstractReward
from costar_task_search.models import GMM

'''
In this case, we want to learn a controller that sticks as close as possible
to the demonstrated trajectory.

We load in a demonstration or a set of demonstrations, and then we use the
distance to the mean path as a cost function during our learning process.
In general, this cost function is:
  R(x, u, t) = - log p (x, u | mu, Sigma, t)
'''
class DemoReward(AbstractReward):

  '''
  Filenames need to be ROS bags
  '''
  def __init__(self, gmm=None, dataset=None, *args, **kwargs):
    if gmm:
      self.gmm = gmm
    elif dataset:
      pass
      self.gmm = None
    else:
      raise RuntimeError("For a demonstration reward, you must" + \
          " provide either a model or a dataset to learn that model" + \
          " from.")

  '''
  Look at the features for this possible world and determine how likely they
  they seem to be under our given feature model.
  '''
  def __call__(self, world):
    # get world features for this state
    f = world.initial_features
    
