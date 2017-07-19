
from abstract import AbstractTaskDefinition

import pybullet as pb


class RingsTaskDefinition(AbstractTaskDefinition):

    '''
    Define a simple task. The robot needs to pick up and manipulate rings.
    '''

    joint_positions = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]

    def __init__(self, *args, **kwargs):
        '''
        Read in arguments for rings. Create a certain number of pegs, some with
        rings on them, some without. Task is to move rings to a particular peg.
        '''
        super(RingsTaskDefinition, self).__init__(*args, **kwargs)

    def _setup(self):
        '''
        Create task by adding objects to the scene
        '''
        pass

    def _setupRobot(self, handle):
        self.robot.place([0, 0, 0], [0, 0, 0, 1], self.joint_positions)
        self.robot.arm(self.joint_positions, pb.POSITION_CONTROL)
        self.robot.gripper(0, pb.POSITION_CONTROL)

    def getName(self):
        return "rings"
