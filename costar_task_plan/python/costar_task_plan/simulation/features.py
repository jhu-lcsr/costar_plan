from costar_task_plan.abstract.features import AbstractFeatures

import numpy as np


def GetAvailableFeatures():
    return ['empty', 'null', 'depth', 'rgb', 'joint_state', 'multi']


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
            'joint_state': JointStateFeatures(),
            'rgb': RgbImageFeatures(),
            'multi': ImagePlusFeatures(),
        }[features]
    except KeyError, e:
        raise NotImplementedError(
            'Feature function %s not implemented!' % task)


class EmptyFeatures(AbstractFeatures):

    '''
    This is a very simple set of features. It does, well, nothing at all. It is
    super fast, though, which makes it good for running execution tests.
    '''

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


class ImagePlusFeatures(AbstractFeatures):

    '''
    Include arm, state, gripper. This set of features is probably the fullest
    representation of the robot's state.
    '''

    def compute(self, world, state):
        return world.cameras[0].capture().rgb, state.arm, state.gripper

    @property
    def description(self):
        return "features", "arm", "gripper"
