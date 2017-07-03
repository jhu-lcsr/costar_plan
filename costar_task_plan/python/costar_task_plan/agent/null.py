from abstract import AbstractAgent

import gym; from gym import spaces
import numpy as np

class NullAgent(AbstractAgent):

    name = "random"

    def __init__(self, env, iter=10000, *args, **kwargs):
        super(NullAgent, self).__init__(*args, **kwargs)
        self.iter = iter
        self.env = env

    def fit(self):
        for i in xrange(10000):
            if isinstance(self.env.action_space, spaces.Tuple):
                cmd = ()
                for space in self.env.action_space.spaces:
                    cmd += (np.zeros(space.shape),)
            else:
                cmd = np.zeros(space.shape)
            self.env.step(cmd)

        return None
        
