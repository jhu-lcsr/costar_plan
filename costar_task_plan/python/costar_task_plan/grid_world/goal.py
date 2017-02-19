from world import *

'''
Each goal is associated with a set of states that are legal.
- states: valid set of states in the world
- next_goals: transitions to possible goals for the task
'''


class Goal:

    def __init__(name, world, states=None, children=None, parents=None):
        self.name = name
        self.world = world

        if children is None:
            self.children = {}
        else:
            self.children = children

        if parents is None:
            self.parents = {}
        else:
            self.parents = parents

        # the set of states associated with this goal
        if states is None:
            self.states = []
        else:
            self.states = states

    '''
    add a child
    this actually creates a two-way connection between child and parent
    '''

    def addChild(self, goal):
        if goal.name not in self.children:
            self.children[goal.name] = goal
        if self.name not in goal.parents:
            goal.parents[self.name] = self
