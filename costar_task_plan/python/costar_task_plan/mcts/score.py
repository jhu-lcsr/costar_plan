
from abstract import *

'''
This is the classic Upper Confidence Bound for Trees score.
'''


class Ucb1Score(AbstractScore):

    def __init__(self, c=1.0):
        self.c = c

    def __call__(self, parent, child):
        if child.n_visits == 0:
            return float('inf')
        else:
            return child.avg_reward + self.c * np.sqrt(np.log(parent.n_visits) / child.n_visits)

'''
This is the "AlphaGo" score.
'''


class PriorProbabilityScore(AbstractScore):

    def __init__(self, c=1.0):
        self.c = c

    def __call__(self, parent, child):
        return child.avg_reward + self.c * child.prior / (1 + child.n_visits)
