
from costar_task_plan.simulation.world import *

from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf_conversions import posemath as pm
from urdf_parser_py.urdf import URDF

import gym; from gym import spaces
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
        self.action_space = self.getActionSpace()

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
        '''
        The ik() command is used by various agents and problem domains
        to recover a command vector that will move the robot arm to the right
        pose.

        Params:
        --------
        pose: kdl frame to move to
        q0: current joint position
        '''
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
        '''
        Returns:
        --------
        q: joint positions
        dq: joint velocities
        '''
        raise NotImplementedError('get joints')

    def _getGripper(self):
        raise NotImplementedError('get gripper')

    def getState(self):
        '''
        Simple tool: take the current simulation and get a state representing
        what the robot will look like.
        '''
        (pos, rot) = pb.getBasePositionAndOrientation(self.handle)
        # TODO(cpaxton): improve forward kinematics efficiency by just using
        # PyBullet to get the position of the grasp frame.
        q, dq = self._getArmPosition()
        return SimulationRobotState(robot=self,
                                    base_pos=pos,
                                    base_rot=rot,
                                    arm=q,
                                    arm_v=dq,
                                    gripper=self._getGripper(),
                                    T=self.fwd(q))

    def command(self, action):
        '''
        Process an incoming action and apply it to the simulated robot. Does
        not currently support robots with mobile bases, but it could.
        '''
        if action.arm_cmd is not None:
            self.arm(action.arm_cmd)
        #else:
        #    self.arm(self._getArmPosition())
        if action.gripper_cmd is not None:
            self.gripper(action.gripper_cmd)
        else:
            self.gripper(self._getGripper())

    def toParams(self, action):
        '''
        Convert action into a reasonable format so that we can save it. Note
        that this assumes that the action is specified as a position to move
        to; if it is not, then you'll need to override this.

        Params:
        -------
        action: a CTP action containing commands for arm, gripper, base
        '''
        if action.arm_cmd is not None:
            arm = action.arm_cmd
        else:
            arm = self._getArmPosition()
        if action.gripper_cmd is not None:
            gripper = action.gripper_cmd
        else:
            gripper = self._getGripper()
        
        return arm, gripper

    def getActionSpace(self):
        '''
        Gives spaces for arm gripper etc
        '''
        raise NotImplementedError('should set up the appropriate spaces' + \
                ' for the robot as a tuple.')

