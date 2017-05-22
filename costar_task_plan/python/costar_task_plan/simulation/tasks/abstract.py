from costar_task_plan.abstract import NullReward
from costar_task_plan.simulation.world import *

import pybullet as pb
import rospkg
import os

class AbstractTaskDefinition(object):

    '''
    Defines how we create the world, gives us the reward function, etc. This
    builds off of the costar task plan library: it will allow us to create a
    World object that represents a particular environment, based off of the
    basic BulletWorld.
    '''

    def __init__(self, robot, seed=None, option=None, *args, **kwargs):
        '''
        We do not create a world here, but we may need to cache things or read
        them off of the ROS parameter server as necessary.
        '''
        self.seed = seed
        self.option = option
        self.robot = robot
        self.world = None

    def setup(self):
        '''
        Create task by adding objects to the scene, including the robot.
        '''
        rospack = rospkg.RosPack()
        path = rospack.get_path('costar_simulation')
        static_plane_path = os.path.join(path,'meshes','world','plane.urdf')
        pb.loadURDF(static_plane_path)

        self.world = SimulationWorld()
        self._setup()
        handle = self.robot.load()
        pb.setGravity(0,0,-9.807)
        for i in range(1000):
            pb.stepSimulation()
        self._setupRobot(handle)

        world.addActor(SimulationRobotActor(robot=self.robot))

    def reset(self):
        '''
        Reset the whole simulation into a working configuration.
        '''
        raise NotImplementedError('Task must override the reset() function!')

    def getReward(self):
        return NullReward()

    def _setup(self):
        '''
        Setup any world objects after the robot has been created.
        '''
        raise NotImplementedError('Must override the _setup() function!')

    def _setupRobot(self, handle):
        '''
        Do anything you need to do to the robot before it
        '''
        raise NotImplementedError('Must override the _setupRobot() function!')

    def cloneRobot(self):
        robot_type = type(self.robot)
        return robot_type()
