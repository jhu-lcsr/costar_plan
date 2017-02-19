
class AbstractFeatures(object):
  def __call__(self,world,state):
    return self.compute(world,state)

  def compute(self, world, state):
    raise Exception("features.compute not yet implemented!")

  def updateBounds(self, world):
    raise Exception('feature.updateBounds not yet implemented!')

  def getBounds(self):
    raise Exception('feature.getBounds not yet implemented!')
