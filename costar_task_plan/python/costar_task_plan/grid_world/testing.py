import world
import copy
from world import *
from actor import *
from experience import ExperienceInput

'''
Was that move OK?
'''


def Evaluate(world, actor, i=0):
    loc = actor.state.lookup(world)
    for other in world.actors:
        if other == actor:
            continue
        elif other.state.x == actor.state.x and other.state.y == actor.state.y:
            # print "COLLISION WITH OTHER ACTOR!"
            return (-1, "collision with actor %s" % (other.name))
    if actor.state.lookup(world) == Intersection:
        if not world.intersection.next_up() == actor.name:
            # print "NOT YOUR TURN!"
            return (-1, "illegal intersection entrance")
    elif not SameDirection(actor.state.theta, world.worldmap[actor.state.y, actor.state.x]):
        # print "WRONG DIRECTION! %d"%actor.state.lookup(world)
        return (-1, "illegal square")

    for goal in actor.goals:
        if actor.state.x == goal.x and actor.state.y == goal.y and actor.state.theta == goal.theta:
            # print "DONE"
            return (1, "finished loop")

    return (0, "not finished")

'''
Evaluate default actor
'''


def EvaluateDefaultActor(world, actor, niter=100):

    # iterate: return 1 if back at spot, else return -1 at end
    # return -1 if it ever leaves its lane or hits another actor too
    for i in xrange(niter):
        # prev_f = actor.getLocalWorld()
        world.tick()
        (code, res) = Evaluate(world, actor, i)
        # print
        # "x=%d,y=%d,location=%d,action=%d"%(actor.state.x,actor.state.y,actor.state.lookup(world),actor.last_action.idx)
        if not code == 0:
            return (code, res)

    return (0, "did not finish loop")

'''
Evaluate default actor
'''


def EvaluateAndGetFeatures(world, actor, num_features, num_actions, get_features, niter=100, default_r=0):

    prev_fs = np.zeros((niter, num_features))
    next_fs = np.zeros((niter, num_features))
    actions = np.zeros((niter, num_actions))
    rs = np.zeros((niter, 1))
    terminal = np.zeros((niter, 1), dtype=bool)

    (code, res) = (-1, "did not finish loop")

    # iterate: return 1 if back at spot, else return -1 at end
    # return -1 if it ever leaves its lane or hits another actor too
    for i in xrange(niter):
        prev_f = get_features(world, actor)
        world.tick()
        next_f = get_features(world, actor)
        (code, res) = Evaluate(world, actor, i)
        # print
        # "x=%d,y=%d,location=%d,action=%d"%(actor.state.x,actor.state.y,actor.state.lookup(world),actor.last_action.idx)

        prev_fs[i] = prev_f
        next_fs[i] = next_f
        actions[i, actor.last_action.idx] = 1.

        rs[i] = default_r

        if not code == 0:
            rs[i] = code
            terminal[i] = True
            break

    if code == 0:
        code = -1
    return (code, res, ExperienceInput(i + 1, prev_fs, rs, actions, next_fs, terminal))
