
"""
(c) 2016 Chris Paxton
"""

import numpy as np
import copy
from Queue import deque

Empty = 0
DirectionEast = 1
DirectionWest = 2
DirectionNorth = 3
DirectionSouth = 4
Intersection = 5
Sidewalk = 6
Shoulder = 7


class Status:
    ok = "ok"
    collision = "collision"


class State(object):

    def __init__(self, x, y, theta, t):
        self.x = x
        self.y = y
        self.theta = theta
        self.t = t
        self.status = Status.ok

    def lookup(self, world):
        return world.worldmap[self.y, self.x]

    def checkCollision(self, world):
        for actor in world.actors:
            if self.x == actor.state.x and self.y == actor.state.y:
                return True
        return False

    def equals(self, state):
        try:
            return self.x == state.x and self.y == state.y and self.theta == state.theta
        except TypeError:
            return False


class Action(object):

    def __init__(self, dx, dtheta, idx=-1):
        self.dx = dx
        self.dtheta = dtheta
        self.cost = 0
        self.idx = idx
        self.total_movement = 0
        self.total_rotation = 0
        self.at_intersection = False
        self.leaving_intersection = False
        self.update()

    def printInfo(self):
        print "dx=%s dtheta=%s c=%f tm=%f tr=%f" % (str(self.dx), str(self.dtheta), self.cost, self.total_movement, self.total_rotation)

    '''
    update internal variables
    '''

    def update(self):
        self.cost = 0
        for (dx, dtheta) in zip(self.dx, self.dtheta):
            self.cost += abs(dx) + abs(dtheta)
            self.total_movement += abs(dx)
            self.total_rotation += abs(dtheta)

    '''
    take some action and apply it to get a new state
    '''

    def apply(self, world, state):

        state = copy.deepcopy(state)
        for (dx, dtheta) in zip(self.dx, self.dtheta):
            # move "forward"
            if state.theta == 0:
                state.y -= dx
            elif state.theta == 1:
                state.x -= dx
            elif state.theta == 2:
                state.y += dx
            elif state.theta == 3:
                state.x += dx
            state.t += 1
            state.theta += dtheta

            state.theta %= 4

            if state.x < 0:
                state.x = world.max_x
            if state.y < 0:
                state.y = world.max_y
            if state.x > world.max_x:
                state.x = 0
            if state.y > world.max_y:
                state.y = 0

        return state, Status.ok

    '''
    return total amount of moves made
    '''

    def cost(self, state):
        cost = 0
        for (dx, dtheta) in zip(self.dx, self.dtheta):
            cost += dx + dtheta
        return cost

'''
IntersectionManager
An important part of the world!
Each car can only go after stopping once, go in order
'''


class IntersectionManager(object):

    def __init__(self):
        self.q = deque()

    def add(self, actor):
        self.q.append(actor)

    def pop(self):
        return self.q.popleft()

    def remove(self, name):
        self.q = deque([a for a in self.q if not a.name == name])

    def next_up(self):
        if len(self.q) > 0:
            return self.q[0].name
        else:
            return ''

'''
World
describes a gridworld containing actors and other objects
states are discretized (x,y,theta).
x and y are on the grid.
theta is on (0,1,2,3)
'''


class GridWorld(object):

    def __init__(self,
                 worldmap=np.array([0]),
                 actors=None
                 ):

        self.worldmap = worldmap
        if not actors is None:
            self.actors = actors
        else:
            self.actors = []
        self.intersection = IntersectionManager()
        self.cmap = {}

        self.max_x = worldmap.shape[1] - 1
        self.max_y = worldmap.shape[0] - 1

        self.x_offset = 0
        self.y_offset = 0

    def draw(self):
        pass

    def removeActorsByName(self, name):
        self.actors = [
            actor for actor in self.actors if not actor.name == name]
        return self

    '''
    return a whole list of actions from the world at a state
    '''

    def getActions(self, state=None):

        # start with a new list of actions
        actions = [
            Action([0, 1], [-1, 0], 0),
            Action([0, 1], [1, 0], 1),
            Action([1], [0], 2),
            Action([0], [0], 3),
            Action([0, 1, 0], [-1, 0, 1], 4),
            Action([0, 1, 0], [1, 0, -1], 5),
        ]

        return actions

    '''
    add actor to the world's list of managed actors
    '''

    def addActor(self, actor):
        actor.setActions(self.getActions())
        self.actors.append(actor)

    def checkCollision(self, state):
        return self.cmap.has_key((state.x, state.y))

    '''
    update the world by calling all actors and getting their moves
    '''

    def tick(self):

        actions = []
        for actor in self.actors:
            actions.append(actor.chooseAction(self))

        self.cmap = {}
        for (actor, action) in zip(self.actors, actions):
            prev_state = actor.state
            (actor.state, status) = action.apply(self, actor.state)
            actor.last_action = action
            self.cmap[(actor.state.x, actor.state.y)] = actor.name

            if prev_state.lookup(self) == Intersection and not actor.state.lookup(self) == Intersection:
                self.intersection.remove(actor.name)
            elif actor.state.equals(prev_state):
                # if near intersection add to queue
                (ahead, ahead_status) = Action(
                    [1], [0], 2).apply(self, actor.state)
                if ahead.lookup(self) == Intersection:
                    self.intersection.add(actor)
            # elif action.at_intersection:
            #    self.intersection.add(actor)

            if not status is Status.ok:
                return status

    '''
    return a copy of the world centered on current position
    '''

    def getLocalWorld(self, actor, horizon=1, includeIntersectionQueue=True):
        localmap = np.zeros(((2 * horizon) + 1, (2 * horizon) + 1))

        for i in range(-1 * horizon, horizon + 1):
            for j in range(-1 * horizon, horizon + 1):
                y = (actor.state.y + j) % self.worldmap.shape[0]
                x = (actor.state.x + i) % self.worldmap.shape[1]

                # print "%f %f %s"%(x,y,str(self.worldmap[y,x]))
                localmap[j + horizon, i + horizon] = self.worldmap[y, x]

        newWorld = GridWorld(worldmap=localmap)
        newWorld.intersection = copy.copy(self.intersection)
        newWorld.actors = copy.copy(self.actors)
        newWorld.x_offset = actor.state.x - horizon
        newWorld.y_offset = actor.state.y - horizon

        return newWorld

    '''
    convert this entire world into features
    Features for each cell:
    - is same direction lane
    - is other direction lane
    - is sidewalk
    - is intersection
    - is occupied by another actor
    '''

    def getFeatures(self, actor, useIntersection=True, flattened=True):

        feature_dim = 5

        features = np.zeros(
            (self.worldmap.shape[0], self.worldmap.shape[1], feature_dim))

        for i in range(self.worldmap.shape[0]):
            for j in range(self.worldmap.shape[1]):

                # check to see if this cell is the same direction
                # check to see if it is

                if SameDirection(actor.state.theta, self.worldmap[i, j]):
                    features[i, j, 0] = 1
                    # road and same direction
                elif self.worldmap[i, j] in [DirectionEast, DirectionWest, DirectionNorth, DirectionSouth]:
                    features[i, j, 1] = 1
                        # different direction but still road
                elif self.worldmap[i, j] == Intersection:
                    features[i, j, 2] = 1
                elif self.worldmap[i, j] == Sidewalk:
                    features[i, j, 3] = 1

                for otherActor in self.actors:
                    if otherActor.name == actor.name:
                        continue

                    if otherActor.state.x == j + self.x_offset and otherActor.state.y == i + self.y_offset:
                        features[i, j, 4] = 1
                        break

        if flattened:
            data = features.flatten().tolist()
            if useIntersection:
                # print self.intersection.next_up()
                data.append(float(self.intersection.next_up() == actor.name))
                # print data

            return np.array(data)

        else:
            return features

'''
Create a map consisting of a single horizontal stretch of road
'''


def HorizontalRoadMap(lanes_per_direction=2, sidewalk=False, length=10, oneway=False):
    arr = []
    height = lanes_per_direction

    if sidewalk:
        arr += [Sidewalk] * length
        height += 2

    if not oneway:
        arr += [DirectionWest] * length * lanes_per_direction
        height += lanes_per_direction

    arr += [DirectionEast] * length * lanes_per_direction

    if sidewalk:
        arr += [Sidewalk] * length

    worldmap = np.array(arr).reshape(height, length)

    return worldmap

'''
do this actor's angle and the direction of the road point the same way?
'''


def SameDirection(angle, direction):
    return (angle == 3 and direction == DirectionEast) \
        or (angle == 1 and direction == DirectionWest) \
        or (angle == 0 and direction == DirectionNorth) \
        or (angle == 2 and direction == DirectionSouth)
