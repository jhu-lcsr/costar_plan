
import copy
import matplotlib.pyplot as plt

from oracle import *

'''
This oracle produces new samples on a set of environments by following some
(presumably optimal) policy.
'''
class PolicyOracle(AbstractOracle):

  def __init__(self, policy, adversary, worlds=100, visualize=False, *args, **kwargs):
    super(PolicyOracle, self).__init__(*args ,**kwargs)
    self.policy = policy
    self.adversary = adversary
    self.worlds = []
    self._data = []
    self.n_worlds = worlds

  '''
  Create a bunch of hard problems
  '''
  def compile(self, *args, **kwargs):
    self.compiled = True

    if len(self._data) == self.n_worlds:
      # we have already done this
      return

    print "Generating training data...",
    for i in xrange(self.n_worlds):
      world = self.adversary.makeWorld(*args, **kwargs)
      orig_world = copy.deepcopy(world)
      traj = []
      
      # solve the world with the policy
      done = False
      F0 = world.computeFeatures()
      i = 0
      while not done:
        actor = world.actors[0]
        state = actor.state
        a = self.policy.evaluate(world, state, actor)
        (res, S0, A0, S1, F1, r) = world.tick(a)
        traj.append(Sample(S0, F0, A0, S1, F1, r))
        F0 = F1

        done = not res

      self._data.append(traj)
      self.worlds.append(orig_world)

    print "done."

  def show(self):
    if not self.compiled:
      raise RuntimeError('compile before trying to draw')

    x = []
    y = []
    v = []
    theta = []
    dw = []
    for traj in self._data:
      for sample in traj:
        # convert from traj to np array and print it?
        x.append(sample.s0.fx)
        y.append(sample.s0.fy)
        theta.append(sample.s0.w)
        v.append(sample.s0.v)
        dw.append(sample.a0.dw)
      
    plt.subplot(2,2,1)
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(2,2,2)
    plt.plot(x,v)
    plt.xlabel('x')
    plt.ylabel('v')
    plt.subplot(2,2,3)
    plt.plot(x,theta)
    plt.xlabel('x')
    plt.ylabel('theta')
    plt.subplot(2,2,4)
    plt.plot(x,dw)
    plt.xlabel('x')
    plt.ylabel('sar')
    plt.show()

  def getWorld(self, i):
    return self.worlds[i]

  def getTrajectory(self, i):
    return self.samples[i]

  def samples(self):
    return self._data

