from abstract import AbstractAgent


class RandomAgent(AbstractAgent):
    '''
    Really simple test agent that just generates a random set of positions to
    move to.
    '''

    name = "random"

    def __init__(self, iter=10000, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
        self.iter = iter

    def fit(self, env):
        for _ in xrange(self.iter):
            cmd = env.action_space.sample()
            env.step(cmd)

        return None
        
