
import numpy as np

from costar_task_plan.abstract import AbstractDynamics

from dynamics_gmm import DynamicsPriorGMM

'''
Model that fits a GMM to a particular sample
'''
class GmmDynamics(AbstractDynamics):

  def __init__(self, traj, hyperparameters):

    self.dynamics_prior = DynamicsPriorGMM()
    self.update(traj)

  def update(self, traj):

    X = []
    U = []
    for sample in traj:
      X.append(sample.f0)
      U.append(sample.a0.toArray())
      #sa = np.append(sample.f0, sample.a0.toArray())
      #sa0_size = sa.shape[0]
      #sas = np.append(sa, sample.f1)
      #data.append(sas)

    #data = np.array(data)
    #self.gmm.update(data, self.K)
    self.dynamics_prior.update(np.array([X]),np.array([U]))


  '''
  NOTE: since this requires features, it's gonna fail if you're calling for a
  non-hero actor.
  '''
  def __call__(self,state,action,dt):
    print state.world
    print state.world.initial_features
    new_f0 = np.append(state.world.initial_features, action.toArray())
