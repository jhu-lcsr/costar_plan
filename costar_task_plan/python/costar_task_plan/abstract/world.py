import copy

from state import AbstractState
from action import AbstractAction

'''
Nonspecific implementation that encapsulates a particular RL/planning problem.
It contains actors, condition, and a reward function.
In particular, it points to one actor called the LEARNER, who we care about.
'''
class AbstractWorld(object):

  # This is fixed: it's the actor that is doing learning and planning in the
  # world.
  learner = 0

  def __init__(self, reward, verbose=False):
    self.reward = reward
    self.verbose = verbose
    self.actors = []
    self.conditions = []
    self.features = None
    self.initial_features = None
    self.initial_reward = 0
    self.done = False
    self.predicates = []
    self.predicate_idx = {}
    self.num_actors = 0
    self.sprites = []
    self.task = None
    self.dt = 0.1
    self.ticks = 0
    self.max_ticks = 100
    self.scale_reward_to_max_ticks = False

    # This variable allows us to change the draw mode to show actors'
    # boxes instead of actors' circles.
    # This should be coupled with new collision conditions.
    self.draw_sprites = False

  def setTask(self, task):
    self.task = task

  def addActor(self, actor):
    actor.setId(len(self.actors))
    actor.state.updatePredicates(self, actor)
    self.actors.append(actor)
    self.num_actors = len(self.actors)

  def addCondition(self, condition, weight, name):
    self.conditions.append((condition, weight, name))

  # override this if there's some cleanup logic that needs to happen after dynamics updates
  def hook(self):
    pass

  def getFeatureBounds(self):
    return self.features.getBounds()

  def setFeatures(self, features):
    features.updateBounds(self)
    self.features = features

  def addPredicate(self, predicate, predicate_check):
    idx = len(self.predicates)
    self.predicate_idx[predicate] = idx
    predicate_check.setIndex(idx)
    self.predicates.append((predicate, predicate_check))

  '''
  For use in planning:
  - take an action
  - take an optional set of policies

  Create a copy of the world and tick() with the appropriate new action.

  If we have policies, actors will be reset appropriately to use new policies.
  '''
  def fork(self, action, policies={}):
    new_world = copy.copy(self)
    # NOTE: this is where the inefficiency lies.
    #new_world.actors = copy.deepcopy(self.actors)
    new_world.actors = [copy.copy(actor) for actor in self.actors]

    (res, S0, A0, S1, F1, r) = new_world.tick(action)
    return new_world

  '''
  Main update loop for the World class.
  
  This resolves all the logic in the world by updating each actor according to
  its last observation of the world state.
  '''
  def tick(self, A0):
    self.ticks += 1

    S0 = self.actors[0].state

    # track list of actions for all entities
    actions = [0]*len(self.actors)
    for i, actor in enumerate(self.actors):
      if i is 0:
        actions[i] = A0
      else:
        actions[i] = actor.evaluate(self)

    # update all actors in a separate loop
    for actor, action in zip(self.actors, actions):
      s = actor.update(action, self.dt)
      actor.state.predicates = [0] * len(self.predicates)
    S1 = self.actors[0].state

    self.hook() # run update hook for this environment
   
    # update all actors
    #self.predicates = [check(world, self, actor, actor.last_state)
    #for actor in self.actors:
    #  actor.state.updatePredicates(self, actor)
    for i, (name, check) in enumerate(self.predicates):
      for actor in self.actors:
        actor.state.predicates[i] = check(
            self,
            actor.state,
            actor,
            actor.last_state)

    (res, F1, r, rt) = self._process() # get the final set of variables

    if not res and self.scale_reward_to_max_ticks:
        # compute difference here
        d_ticks = max(0, self.max_ticks - self.ticks)
        r *= 1 + d_ticks

    # update features and reward
    self.initial_features = F1
    self.initial_reward = r + rt
    self.done = not res

    return (res, S0, A0, S1, F1, r + rt)

  def _process(self):
    # compute features
    if self.features is not None:
      # NOTE: this is about 0.1 second right now in larger trees
      F1 = self.computeFeatures()
    else:
      F1 = None
    r, bonus = self.reward(self)
    rt = bonus

    # determine if we should terminate
    res = True
    actor = self.actors[0]
    state = actor.state
    prev_state = actor.last_state
    for (condition, weight, name) in self.conditions:
      if not condition(self, state, actor, prev_state):
        if self.verbose:
          print "Constraint violated: %s, score modifier: %f"%(name,weight)
        res = False
        rt = rt + weight

    return res, F1, r, rt

  '''
  May be overridden in some cases
  '''
  def computeFeatures(self):
    return self.features(self, self.actors[0].state)

  def updateFeatures(self):
    self.initial_features = self.computeFeatures()

  '''
  Destroy all current actors.
  '''
  def reset(self):
    self.actors = []
    self._reset()

  '''
  Specify graphics to use when drawing this world.
  '''
  def setSprites(self, sprites):
    self.sprites = sprites
    self.draw_sprites = len(sprites) > 0

  '''
  Clear any graphics used to draw this world.
  '''
  def clearSprites(self):
    self.sprites = []
    self.draw_sprites = False

  def debugPredicates(self, state):
    for p, (n, c) in zip(state.predicates, self.predicates):
      print n,"=",p,","
    print ""

  '''
  This is the "player" whose moves we want to optimize.
  '''
  def getLearner(self):
    return self.actors[0]

