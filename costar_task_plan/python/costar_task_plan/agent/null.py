from abstract import AbstractAgent

import gym; from gym import spaces
import numpy as np

class NullAgent(AbstractAgent):

    name = "random"

    def __init__(self, iter=10000, *args, **kwargs):
        super(NullAgent, self).__init__(*args, **kwargs)
        self.iter = iter

    def fit(self, env):
        for i in xrange(10000):
            if isinstance(env.action_space, spaces.Tuple):
                cmd = ()
                for space in env.action_space.spaces:
                    cmd += (np.zeros(space.shape),)
            else:
                cmd = np.zeros(space.shape)
            env.step(cmd)

        return None
        
