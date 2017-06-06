
from costar_task_plan.simulation.world import *

import pybullet as pb


class AbstractRobotInterface(object):

    '''
    This defines the functions needed to send commands to a simulated robot,
    whatever that robot might be. It should check values then call the
    appropriate PyBullet functions to send over to the server.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Parse through the config, save anything that needs to be saved, and
        initialize anything that needs to be initialized. May connect to ROS
        for additional parameters.
        '''
        pass

    def load(self):
        '''
        This function should take the robot, load it from file somehow, and put
        that model into the simulation.
        '''
        raise NotImplementedError(
            'This has to put the robot into the simulation.')

    def place(self, pos, joints):
        '''
        Update the robot's position.
        '''
        raise NotImplementedError('This should put the robot in a specific pose.')

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
        raise NotImplementedError('The inverse() command takes a position' + \
                                  'and gets inverse kinematics associated' + \
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
