
"""
(c) 2016 Chris Paxton
"""

from task_tree_search.abstract import *
from world import *

'''
The Actor class describes anything that can move and interact with the world in a particular way.
It will respond to a supposition of the world at a particular point in time.

NOTE: this is not fully integrated with the new API.
'''
class Actor(AbstractActor):

    def __init__(self, state=State(0, 0, 0, 0),
                 name="",
                 x=0., y=0., theta=0.):

        self.name = name
        self.state = state
        self.x = x
        self.y = y
        self.theta = theta
        self.at_intersection = False
        self.waited = False
        self.actions = []
        self.last_action = None
        self.goals = []

    def move(self):
        pass

    def setGoals(self, goals):
        self.goals = goals

    '''
    update this actor's current state
    '''

    def setState(self, state):
        self.state = state

    def getState(self):
        return self.state

    def evaluate(self, world):
      return chooseAction(self, world)

    def chooseAction(self, world):
        return None

    def reward(self, state, world):
        pass

    '''
    called after each motion
    '''

    def cleanup(self):
        pass

    '''
    provide the possible actions in the given world
    '''

    def setActions(self, actions):
        self.actions = actions

'''
The default actor stays in its lane and continues in the appropriate direction forever.
'''


class DefaultActor(Actor):

    def __init__(self, state,
                 name="D"):
        super(DefaultActor, self).__init__(state, name)
        self.impatience = 0

    def chooseAction(self, world):

        world = copy.copy(world).removeActorsByName(self.name)
        # world.tick()

        best_action = None
        ba_rotation = 1
        ba_fwd = 1
        no_motion = None
        nm_cost = 9999
        at_intersection = False

        for action in self.actions:
            if action.cost < nm_cost:
                no_motion = action
                nm_cost = action.cost
            if action.total_rotation <= ba_rotation and action.total_movement >= ba_fwd:
                new_state, status = action.apply(world, self.state)

                if not status == Status.ok or world.checkCollision(new_state):
                    continue
                next_space = new_state.lookup(world)
                if not (next_space == Intersection or SameDirection(new_state.theta, next_space)):
                    continue
                elif next_space == Intersection:
                    at_intersection = True

                if not next_space == Intersection or world.intersection.next_up() == self.name:
                    ba_fwd = action.total_movement
                    ba_rotation = action.total_rotation
                    best_action = action

        if not best_action is None:
            self.impatience = 0
            return best_action
        else:
            self.impatience += 1
            no_motion.at_intersection = at_intersection
            return no_motion

'''
The static actor just remains at its current location indefinitely
'''


class StaticActor(Actor):

    def __init__(self, state,
                 name="StaticActor"):
        super(StaticActor, self).__init__(state, name)

    def chooseAction(self, world):

        no_motion = None
        nm_cost = 9999

        for action in self.actions:
            if action.cost < nm_cost:
                no_motion = action
                nm_cost = action.cost

        return no_motion

'''
The random actor chooses moves at random
'''


class RandomActor(Actor):

    def __init__(self, state,
                 name="r"):
        super(RandomActor, self).__init__(state, name)

    def chooseAction(self, world):
        return self.actions[np.random.randint(len(self.actions))]
