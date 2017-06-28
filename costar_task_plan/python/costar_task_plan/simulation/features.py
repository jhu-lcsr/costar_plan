from costar_task_plan.abstract.features import AbstractFeatures

import numpy as np

def GetAvailableFeatures():
    return ['null', 'depth']

def GetFeatures(features):
    '''
    Returns a particular task definition in the simulation.
    '''
    try:
        return {
            '': EmptyFeatures(),
            'null': EmptyFeatures(),
            'empty': EmptyFeatures(),
            'depth': DepthImageFeatures(),
            'joint_state' : JointStateFeatures(),
            'rgb': RgbImageFeatures(),
        }[features]
    except KeyError, e:
        raise NotImplementedError('Feature function %s not implemented!' % task)

class EmptyFeatures(AbstractFeatures):
  def compute(self, world, state):
      return np.array([0])

  def updateBounds(self, world):
      pass

  def getBounds(self):
      return np.array([0]), np.array([0])

class DepthImageFeatures(AbstractFeatures):
  '''
  The only features we return are the depths associated with each camera pixel.
  So we get 2.5D data here.
  '''
  def compute(self, world, state):
      return world.cameras[0].capture().depth

  def updateBounds(self, world):
    raise Exception('feature.updateBounds not yet implemented!')

  def getBounds(self):
    raise Exception('feature.getBounds not yet implemented!')



class JointStateFeatures(AbstractFeatures):

  def compute(self, world, state):
      return np.append(state.arm, state.gripper)
      

  def updateBounds(self, world):
    raise Exception('feature.updateBounds not yet implemented!')

  def getBounds(self):
    raise Exception('feature.getBounds not yet implemented!')

class RgbImageFeatures(AbstractFeatures):
  '''
  The only feature data we return will be a single RGB image from the first
  camera placed in the world, where ever that may be.
  '''
  def compute(self, world, state):
      return world.cameras[0].capture().rgb


