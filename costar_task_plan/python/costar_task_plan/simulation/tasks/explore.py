

from abstract import AbstractTaskDefinition

import pybullet as pb


class ExploreTaskDefinition(AbstractTaskDefinition):

    '''
    Robot must move around in the environment and explore an area, complete
    with clutter.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Arguments define how to create a house and populate it with rabdom
        objects.
        '''
        super(ExploreTaskDefinition, self).__init__(*args, **kwargs)

    def _setup(self):
        '''
        Create task by adding objects to the scene.
        '''

    def _setupRobot(self, handle):
        '''
        Robot must be mobile.
        '''
        if not self.robot.mobile():
            raise RuntimeError('Exploration task does not even make sense' \
                               + 'without a mobile robot.')
        self.robot.place([0,0,0],[0,0,0,1],self.joint_positions)
        #self.robot.arm(self.joint_positions, pb.POSITION_CONTROL)
        #self.robot.gripper(0, pb.POSITION_CONTROL)
