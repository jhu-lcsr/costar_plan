import pybullet as pb

from abstract import AbstractRobotInterface

class Ur5RobotiqInterface(AbstractRobotInterface):
    '''
    Defines action space for the ur5 with a robotiq 85 gripper. This is the
    standard "costar" robot used for many of our experiments.
    '''
    
    def __init__(self, *args, **kwargs):
        super(Ur5RobotiqInterface, self).__init__(*args, **kwargs)
