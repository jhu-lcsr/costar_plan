
from abstract import *
from action import *

import numpy as np
import operator


class GpPolicySample(AbstractSample):

    def __init__(self, distributions={}, data={}):
        self.Z = distributions
        self.data = data

    def numOptions(self):
        return 1

    def getOption(self, node, idx):
        return MctsAction(policy=self.policy, id=0, ticks=self.ticks)

    def _sample(self, node):
        return MctsAction(policy=self.policy, id=0, ticks=self.ticks)

    def getPolicies(self, node):
        return [self.policy]

    def getName(self):
