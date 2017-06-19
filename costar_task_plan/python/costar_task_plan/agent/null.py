from abstract import AbstractAgent

import numpy as np

class NullAgent(AbstractAgent):

    name = "random"

    def __init__(self, iter=10000, *args, **kwargs):
        super(NullAgent, self).__init__(*args, **kwargs)
        self.iter = iter

    def fit(self, env):
        for i in xrange(10000):
            cmd = np.zeros(env.action_space,))
            env.step(cmd)

        return None
        
