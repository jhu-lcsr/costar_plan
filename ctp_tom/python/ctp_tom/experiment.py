import rospy

def MakeExperiment(experiment):
    pass

class TomExperiment(object):

    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        '''
        Set up tom simulation scene; this 
        '''
        raise NotImplementedError('reset not implemented')

    def update(self):
        '''
        Placeholder for updating data from the world I guess
        '''
        pass

