import pybullet as pb


class AbstractTaskDefinition(object):

    '''
    Defines how we create the world, gives us the reward function, etc. This
    builds off of the costar task plan library: it will allow us to create a
    World object that represents a particular environment, based off of the
    basic BulletWorld.
    '''

    def __init__(self, robot, seed=None, *args, **kwargs):
        '''
        We do not create a world here, but we may need to cache things or read
        them off of the ROS parameter server as necessary.
        '''
        self.seed = seed
        self.robot = robot

    def setup(self):
        '''
        Create task by adding objects to the scene, including the robot.
        '''
        self._setup()
        handle = self.robot.load()
        self._setupRobot(handle)

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
