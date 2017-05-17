
from abstract import AbstractTaskDefinition

import pybullet as pb


class BlocksTaskDefinition(AbstractTaskDefinition):

    '''
    Define a simple task. The robot needs to pick up and stack blocks of
    different colors in a particular order.
    '''

    joint_positions = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]

    def __init__(self, *args, **kwargs):
        '''
        Read in arguments defining how many blocks to create, where to create
        them, and the size of the blocks. Size is given as mean and covariance,
        blocks are placed at random.
        '''
        super(BlocksTaskDefinition, self).__init__(*args, **kwargs)

    def _setup(self):
        '''
        Create task by adding objects to the scene
        '''
        pass

    def _setupRobot(self, handle):
        for i, q in enumerate(self.joint_positions):
            pb.resetJointState(handle, i, q)
