

class AbstractAgent(object):
    '''
    Default agent. Wraps a large number of different methods for learning a
    neural net model for robot actions.
    '''

    name = None
    
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, env):
        raise NotImplementedError('fit() should run algorithm on the environment')

    def data(self):
        '''
        Returns dataset.
        '''
        raise NotImplementedError('not yet working')
