from abstract import AbstractAgent


class RandomAgent(AbstractAgent):

    name = "random"

    def __init__(self, iter=10000, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
        self.iter = iter

    def fit(self, env):
        for i in xrange(10000):
            cmd = env.action_space.sample()
            env.step(cmd)

        return None
        
