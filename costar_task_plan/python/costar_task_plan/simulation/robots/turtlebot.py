
from abstract import AbstractRobotInterface

import gym
from gym import spaces
import numpy as np
import os
import pybullet as pb
import rospkg
import subprocess
from costar_task_plan.simulation.world import *



class TurtlebotInterface(AbstractRobotInterface):

    '''
    Defines action space for the Turtlebot mobile robot arm.
    '''

    xacro_filename = 'robot/create_circles_kinect.urdf.xacro'
    urdf_filename = 'create_circles_kinect.urdf'

    arm_name = "None"
    gripper_name = "None"
    base_name = "turtlebot"
    
    left_wheel_index = 6
    right_wheel_index = 7

    

    def __init__(self, *args, **kwargs):
        super(TurtlebotInterface, self).__init__(*args, **kwargs)
    
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

        self.handle = pb.loadURDF(urdf_filename)
        self.grasp_idx = self.findGraspFrame()
        #self.loadKinematicsFromURDF(urdf_filename, "base_link")

        return self.handle
        
    def mobile(self):
        return True

    def getState(self):
        (pos, rot) = pb.getBasePositionAndOrientation(self.handle)
        return SimulationRobotState(robot=self,
                                    base_pos=pos,
                                    base_rot=rot)


    def place(self, pos, rot, joints):
        pass    

    def arm(self, cmd, mode=pb.POSITION_CONTROL):
        pass
    
    def gripperCloseCommand(cls):
        '''
        Return the closed position for this gripper.
        '''
        return np.array([0.0])

    def gripperOpenCommand(cls):
        '''
        Return the open command for this gripper
        '''
        return np.array([0.0])
        

    def base(self, cmd):
        maxForce = 500
        #for j in range (0,i):
        
        vr = cmd[0];
        va = cmd[1];
        
        wheel_sep_ = 0.34; # TODO: Verify

        wheel_speed_left = vr - va * (wheel_sep_) / 2;
        wheel_speed_right = vr + va * (wheel_sep_) / 2;
        pb.setJointMotorControl2(self.handle, jointIndex=self.left_wheel_index, controlMode=pb.VELOCITY_CONTROL,targetVelocity = wheel_speed_left,force = maxForce)
        pb.setJointMotorControl2(self.handle, jointIndex=self.right_wheel_index, controlMode=pb.VELOCITY_CONTROL,targetVelocity = wheel_speed_right,force = maxForce)
        

    def gripper(self, cmd, mode=pb.POSITION_CONTROL):
        '''
        Gripper commands need to be mirrored to simulate behavior of the actual
        UR5.
        '''
        pass

    def act(self, action):
        '''
        Parse a list of continuous commands and send it off to the robot.
        '''
        pass

    def getActionSpace(self):
        
        #return spaces.Tuple((spaces.Box(-np.pi, np.pi, self.dof),
        #        spaces.Box(-0.8, 0.0, 1)))        
        
        return spaces.Tuple((spaces.Box(-np.pi, np.pi, 6),
                spaces.Box(-0.8, 0.0, 1),  spaces.Box(-np.pi, np.pi, 2)))
        
    def _getArmPosition(self):
        '''
        Get arm information.

        Returns:
        ---------
        q: vector of joint positions
        dq: vector of joint velocities
        '''
        return np.array(0), np.array(0)

    def _getGripper(self):
        return np.array([0])


    
# -*- coding: utf-8 -*-

