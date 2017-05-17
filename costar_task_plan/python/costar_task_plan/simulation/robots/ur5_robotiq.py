import os
import pybullet as pb
import rospkg
import subprocess

from abstract import AbstractRobotInterface


class Ur5RobotiqInterface(AbstractRobotInterface):

    '''
    Defines action space for the ur5 with a robotiq 85 gripper. This is the
    standard "costar" robot used for many of our experiments.
    '''

    xacro_filename = 'robot/ur5_joint_limited_robot.xacro'
    urdf_filename = 'ur5_joint_limited_robot.urdf'

    def __init__(self, *args, **kwargs):
        super(Ur5RobotiqInterface, self).__init__(*args, **kwargs)

    def load(self):
        '''
        This is an example of a function that allows you to load a robot from
        file based on command line arguments. It just needs to find the
        appropriate directory, use xacro to create a temporary robot urdf,
        and then load that urdf with PyBullet.
        '''

        rospack = rospkg.RosPack()
        path = rospack.get_path('costar_simulation')
        filename = os.path.join(path, self.xacro_filename)
        urdf = open(self.urdf_filename, "w")

        # Recompile the URDF to make sure it's up to date
        subprocess.call(['rosrun', 'xacro', 'xacro.py', filename], stdout=urdf)

        return pb.loadURDF(self.urdf_filename)
