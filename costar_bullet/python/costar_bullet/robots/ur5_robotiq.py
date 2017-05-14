import pybullet as pb

from abstract import AbstractRobotInterface

class Ur5RobotiqInterface(AbstractRobotInterface):
    '''
    Defines action space for the ur5 with a robotiq 85 gripper. This is the
    standard "costar" robot used for many of our experiments.
    '''

    xacro_filename = '.xacro'
    
    def __init__(self, *args, **kwargs):
        super(Ur5RobotiqInterface, self).__init__(*args, **kwargs)

    def load(self):
        '''
        This is an example of a function that allows you to load a robot from
        file based on command line arguments. It just needs to find the
        appropriate directory, use xacro to create a temporary robot urdf,
        and then load that urdf with PyBullet.
        '''

        raise NotImplementedError('this does not work yet')
        subprocess.call(['rosrun','xacro','xacro.py',self.urdf_filename])

        pb.loadURDF('ur5_joint_limited_robot.urdf')
