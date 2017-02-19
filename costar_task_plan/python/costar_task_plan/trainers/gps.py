
from abstract import *
from task_tree_search.models import AbstractOracle

'''
Guided policy search trainer: implementation based on guided policy search/

This approach works by computing an optimal trajectory through an environment.

In our case, we need a way of generating some number of demonstrations. We 
optimize these demonstration trajectories with whatever our favorite optimal
control method is, and then minimize the distance between the outputs of our
learned model and this trajectory.

We term the thing generating our initial demonstrations the "oracle."
'''
class GuidedPolicySearchTrainer(AbstractTrainer):

  def __init__(self, oracle, *args, **kwargs):

    # make sure we actually have a source of data
    if not isinstance(oracle, AbstractOracle):
      raise RuntimeError('illegal argument provided.')

    # the oracle gets our data at each iteration
    # how we get this data is up to you, the user
    # when optimizing the neural net, we want to ensure our similarity to this
    # data
    self.oracle = oracle
    self.T = []


  # during step:
  # 1) get new data
  # 2) fit new dynamics model
  def _step(self, data, *args, **kwargs):

    self.oracle.compile()
    self.T = []
    for sample in oracle.getSamples():
      self.T.append(GmmDynamics(sample))

