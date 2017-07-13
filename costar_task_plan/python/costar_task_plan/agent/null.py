from abstract import AbstractAgent

import gym; from gym import spaces
import numpy as np

class NullAgent(AbstractAgent):

    name = "null"

    def __init__(self, *args, **kwargs):
        super(NullAgent, self).__init__(*args, **kwargs)

    def _fit(self, num_iter):
        return None
        
