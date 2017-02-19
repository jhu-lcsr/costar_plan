import copy
from world import *
from actor import *

'''
Create a hard-coded world with a four-way stop
'''


def GetCrossworld():
    crossworld_raw = [
        "                                #vv^^#                               ",
        "                                #vv^^#                               ",
        "                                #vv^^#                               ",
        "                                #vv^^#                               ",
        "                                #vv^^#                               ",
        "                                #vv^^#                               ",
        "                                #vv^^#                               ",
        "#################################vv^^################################",
        "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<XXXX<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<XXXX<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>XXXX>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>XXXX>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
        "#################################vv^^################################",
        "                                #vv^^#                               ",
        "                                #vv^^#                               ",
        "                                #vv^^#                               ",
        "                                #vv^^#                               ",
        "                                #vv^^#                               ",
    ]

    arr = []
    for row in crossworld_raw:
        new_row = []
        for char in row:
            if char == '>':
                num = DirectionEast
            elif char == '<':
                num = DirectionWest
            elif char == '^':
                num = DirectionNorth
            elif char == 'v':
                num = DirectionSouth
            elif char == 'X':
                num = Intersection
            elif char == '#':
                num = Sidewalk
            elif char == ' ':
                num = Empty
            else:
                num = Empty
            new_row.append(num)
        arr.append(new_row)

    crossworld_map = np.array(arr)

    world = GridWorld(worldmap=crossworld_map)

    return world

'''
Get crossworld goals
'''


def GetCrossworldGoals():
    all_goals = [
        Goal("North", states=[State(35, 0, 0, 0), State(36, 0, 0, 0)]),
        Goal("South", states=[State(33, 18, 0, 0), State(34, 18, 0, 0)]),
        Goal("East", states=[State(), State()]),
        Goal("West", states=[State(), State()]),
        Goals("Intersection", states=[]),
    ]

    goals = {}
    for g in all_goals:
        goals[g.name] = g

    return goals

'''
Generate a random default actor for Crossworld
'''


def GetCrossworldDefaultActor(world=None, name="D", bothLanes=True, edgesOnly=False):

    # try to generate an actor 1000 times
    for i in range(1000):

        direction = np.random.randint(DirectionEast, DirectionEast + 4)
        if direction == DirectionEast:
            if not edgesOnly:
                x = np.random.randint(69)
            else:
                x = 0
            if bothLanes:
                y = np.random.randint(10, 12)
            else:
                y = 11
            rot = 3
            goals = [State(68, 10, 3, 0), State(68, 11, 3, 0)]
        elif direction == DirectionWest:
            if not edgesOnly:
                x = np.random.randint(69)
            else:
                x = 68
            if bothLanes:
                y = np.random.randint(8, 10)
            else:
                y = 8
            rot = 1
            goals = [State(0, 8, 1, 0), State(0, 9, 1, 0)]
        elif direction == DirectionSouth:
            if bothLanes:
                x = np.random.randint(33, 35)
            else:
                x = 33
            if not edgesOnly:
                y = np.random.randint(18)
            else:
                y = 0
            rot = 2
            goals = [State(33, 17, 2, 0), State(34, 17, 2, 0)]
        elif direction == DirectionNorth:
            if bothLanes:
                x = np.random.randint(35, 37)
            else:
                x = 36
            if not edgesOnly:
                y = np.random.randint(18)
            else:
                y = 17
            goals = [State(35, 0, 0, 0), State(36, 0, 0, 0)]

            rot = 0
        else:
            continue

        newActor = DefaultActor(State(x, y, rot, 0), name)
        newActor.setGoals(goals)

        if not GridWorld is None:

            placed = True

            # make sure there is no collision with existing actors
            for actor in world.actors:
                if actor.state.x == newActor.state.x and actor.state.y == newActor.state.y:
                    placed = False
                    break

            if not placed:
                continue
            else:
                break
        else:
            break

    return newActor
