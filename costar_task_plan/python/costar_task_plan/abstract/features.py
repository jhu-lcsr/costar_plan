
class AbstractFeatures(object):
  def __call__(self,world,state):
    return self.compute(world,state)

  def compute(self, world, state):
    raise Exception("features.compute not yet implemented!")

  def updateBounds(self, world):
    raise Exception('feature.updateBounds not yet implemented!')

  def getBounds(self):
    raise Exception('feature.getBounds not yet implemented!')

  def getDescription(self):
      '''
      This should return a tuple describing each of the features we care about
      here. It's just for labeling purposes and to make things easier on the
      user if we have features that return tuples of different types of
      information.
      '''
      return "features"
