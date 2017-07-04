
from costar_task_plan.simulation.world import *

from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf_conversions import posemath as pm
from urdf_parser_py.urdf import URDF

import pybullet as pb


class AbstractRobotInterface(object):

    '''
    This defines the functions needed to send commands to a simulated robot,
    whatever that robot might be. It should check values then call the
    appropriate PyBullet functions to send over to the server.
    '''

    grasp_link = "grasp_link"

    def __init__(self, *args, **kwargs):
        '''
        Parse through the config, save anything that needs to be saved, and
        initialize anything that needs to be initialized. May connect to ROS
        for additional parameters.

        Handle should contain the ID of the robot.
        '''
        self.handle = None
        self.grasp_idx = None
        self.kinematics = None

    def load(self):
        '''
        This function should take the robot, load it from file somehow, and put
        that model into the simulation.
        '''
        raise NotImplementedError(
            'This has to put the robot into the simulation.')

    def findGraspFrame(self):
        '''
        Helper function to look up the grasp frame associated with a robot.
        '''
        joints = pb.getNumJoints(self.handle)
        grasp_idx = None
        for i in xrange(joints):
            idx, name, jtype, qidx, \
                uidx, flags, damping, friction, \
                lower, upper, max_force, max_vel, \
                link_name = pb.getJointInfo(self.handle, i)
            if link_name == self.grasp_link:
                grasp_idx = i
                break
        return grasp_idx

    def loadKinematicsFromURDF(self, filename, base_link):
        '''
        Load KDL kinematics class for easy lookup of posiitons from the urdf
        model of a particular robot.

        Params:
        --------
        filename: absolute path to URDF file
        base_link: root of kinematic tree
        '''
        urdf = URDF.from_xml_file(filename)
        tree = kdl_tree_from_urdf_model(urdf)
        chain = tree.getChain(base_link, self.grasp_link)
        self.kinematics = KDLKinematics(urdf, base_link, self.grasp_link)

    def ik(self, pose, q0):
        return self.kinematics.inverse(pm.toMatrix(pose), q0)

    def forward(self, position):
        return self.kinematics.forward(position)

    def fwd(self, q):
        return pm.fromMatrix(self.kinematics.forward(q))

    def gripperCloseCommand(cls):
        '''
        Return the closed position for this gripper.
        '''
        raise NotImplementedError('This should close the robot gripper.')

    def gripperOpenCommand(cls):
        '''
        Return the open command for this gripper
        '''
        raise NotImplementedError('This should open the robot gripper.')

    def place(self, pos, joints):
        '''
        Update the robot's position.
        '''
        raise NotImplementedError(
            'This should put the robot in a specific pose.')

    def arm(self, cmd, mode):
        '''
        Send a command to the arm.
        '''
        raise NotImplementedError('arm')

    def gripper(self, cmd, mode):
        '''
        Send a command to the gripper.
        '''
        raise NotImplementedError('gripper')

    def base(self, cmd, mode):
        '''
        Send a command to the base.
        '''
        raise NotImplementedError('base')

    def mobile(self):
        '''
        Overload this for a mobile robot like the Husky.
        '''
        return False

    def _getArmPosition(self):
        raise NotImplementedError('get joints')

    def _getGripper(self):
        raise NotImplementedError('get gripper')

    def getState(self):
        '''
        Simple tool: take the current simulation and get a state representing
        what the robot will look like.
        '''
        (pos, rot) = pb.getBasePositionAndOrientation(self.handle)
        return SimulationRobotState(robot=self,
                                    base_pos=pos,
                                    base_rot=rot,
                                    arm=self._getArmPosition(),
                                    gripper=self._getGripper())

    def inverse(self, pose):
        '''
        The inverse() command is used by various agents and problem domains
        to recover a command vector that will move the robot arm to the right
        pose.
        '''
        raise NotImplementedError('The inverse() command takes a position' +
                                  'and gets inverse kinematics associated' +
                                  'with it.')

    def getActionSpace(self):
        '''
        Defines the action space used by the robot.
        '''
        raise NotImplementedError('no getActionSpace() implemented')

    def act(self, action):
        '''
        Parse a robot action. Should call the base(), gripper(), or arm()
        functions to set the appropriate commands.
        '''
        raise NotImplementedError('act')
