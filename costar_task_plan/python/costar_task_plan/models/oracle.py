
class Sample(object):
  def __init__(self, s0, f0, a0, s1, f1, r):
    self.s0 = s0
    self.f0 = f0
    self.a0 = a0
    self.s1 = s1
    self.f1 = f1
    self.r = r

'''
Class for computing the effects of a particular policy on a particular world.
'''
class AbstractOracle(object):

  '''
  Init:
  - take an option
  - use this option to create a number of training worlds somehow
  - produce trajectories for those training worlds
  '''
  def __init__(self, option=None):
    self.option = option
    self.compiled = False

  def compile(self):
    raise RuntimeError('precompute not supported for this oracle yet')

  def getWorld(self, i):
    raise RuntimeError('getting environment not yet implemented')

  def getTrajectory(self, i):
    raise RuntimeError('getting expert trajectory not yet implemented')

  def samples(self):
    raise RuntimeError('sampling environment and expert trajectory' + \
        ' not yet implemented')

