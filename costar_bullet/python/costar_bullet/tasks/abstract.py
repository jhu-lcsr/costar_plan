import pybullet as pb

class AbstractTaskDefinition(object):
    '''
    Defines how we create the world, gives us the reward function, etc. This
    builds off of the costar task plan library: it will allow us to create a 
    World object that represents a particular environment, based off of the
    basic BulletWorld.
    '''
    
    def __init__(self, seed=None, *args, **kwargs):
        '''
        We do not create a world here, but we may need to cache things or read
        them off of the ROS parameter server as necessary.
        '''
        self.seed = seed

    def init(self):
        '''
        Create task by adding objects to the scene
        '''
        raise NotImplementedError('Must override the init() function!')
