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
