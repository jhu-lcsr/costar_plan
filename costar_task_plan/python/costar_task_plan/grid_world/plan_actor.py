
from actor import *

'''
This type of actor moves from one place to another based on its goals.
'''


class PlanActor:

    def __init__(self):
        pass


class HeuristicPlanActor:

    def __init__(self):
        pass

    '''
    Provide the function we will use to decide which actions are better
    '''

    def setHeuristic(self, heuristic):
        self.heuristic = heuristic
