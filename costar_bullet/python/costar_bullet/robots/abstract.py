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


