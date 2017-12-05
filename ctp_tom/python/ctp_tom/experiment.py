import rospy

def GetExperiments():
    return ["fake_oranges"]

def MakeExperiment(experiment):
    return {
        "fake_oranges": FakeOrangesExperiment(*args,**kwargs)
    }[experiment]

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
        raise NotImplementedError('update not implemented')

class FakeOrangesExperiment(TomExperiment):
    def reset(self):
        pass
    def update(self):
        pass

