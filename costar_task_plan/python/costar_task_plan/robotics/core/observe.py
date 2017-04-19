
# Observe() functor that should take a world and update it at a particular
# timestep.
class AbstractObserve(object):

  def __call__(self, world):
    raise NotImplementedError('observe() not implemented. This function should'
                              ' take a single world as an argument and update '
                              'it based on sensor or simulation data.')

  def dynamics(self, world, actor):
    raise NotImplementedError('This function retrieves a set of observation '
                              'dynamics for a given world.')
