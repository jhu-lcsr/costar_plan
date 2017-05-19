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

    left_knuckle = 8
    left_finger = 9
    left_inner_knuckle = 12
    left_fingertip = 13

    right_knuckle = 10
    right_finger = 11
    right_inner_knuckle = 14
    right_fingertip = 15

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
        urdf_filename = os.path.join(path, 'robot', self.urdf_filename)
        urdf = open(urdf_filename, "w")

        # Recompile the URDF to make sure it's up to date
        subprocess.call(['rosrun', 'xacro', 'xacro.py', filename], stdout=urdf)

        self.handle = pb.loadURDF(self.urdf_filename)

        pb.createConstraint(self.handle,-1,-1,-1,pb.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0])

        pb.createConstraint(self.handle, self.left_finger,
                self.handle,self.left_fingertip,
                pb.JOINT_POINT2POINT,
                [0,0,1],[0.05,0,0],[0,0,0])
        pb.createConstraint(self.handle, self.right_finger,
                self.handle,self.right_fingertip,
                pb.JOINT_POINT2POINT,
                [0,0,1],[0.05,0,0],[0,0,0])

        return self.handle


    def arm(self, cmd, mode):
        if len(cmd) > 6:
            raise RuntimeError('too many joint positions')
        for i, q in enumerate(cmd):
            pb.setJointMotorControl2(self.handle, i, mode, q)

    def gripper(self, cmd, mode):
        pb.setJointMotorControl2(self.handle, self.left_knuckle, mode,  cmd)
        pb.setJointMotorControl2(self.handle, self.left_inner_knuckle, mode,  cmd)
        pb.setJointMotorControl2(self.handle, self.right_knuckle, mode,  -cmd)
        pb.setJointMotorControl2(self.handle, self.right_inner_knuckle, mode,  -cmd)
