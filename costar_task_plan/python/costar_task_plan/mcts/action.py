from abstract import *
from task_tree_search.abstract import *


class MctsAction(AbstractMctsAction):

  def __init__(self, policy=None, id=0, ticks=10, condition=None, tag=None, *args,**kwargs):
    super(MctsAction, self).__init__(*args,**kwargs)
    self.policy = policy
    self.ticks_after_fork = ticks - 1
    self.condition = condition
    self.tag = tag
    self.id = id
    if self.ticks_after_fork < 0:
      raise RuntimeError('negative number of ticks does not make sense')

  '''
  Take the current node/world state and compute the action we would get
  following the current policy.
  '''
  def getAction(self, node):
    if self.policy is None:
      raise RuntimeError('must provide non-empty policy to compute actions')
    else:
      return self.policy(world=node.world,
          actor=node.world.getLearner(),
          state=node.world.getLearner().state)

  '''
  Overload apply() with default behavior.
  At each MCTs step, we apply() our Mcts action to the new node.
  '''
  def apply(self, node):
    child = node.expand(self.getAction(node))
    return self.update(child)

  '''
  Simulate a child forward in time.
  '''
  def update(self, node):
    if not node.terminal:
      if self.condition is None:
        for i in xrange(self.ticks_after_fork):
          (res, S0, A0, S1, F1, r) = node.tick(self.getAction(node))
          if not res:
            break
      else:
        while self.condition(node.world,
                               node.state,
                               node.world.actors[0],
                               node.world.actors[0].last_state):
          (res, S0, A0, S1, F1, r) = node.tick(self.getAction(node))
          if not res:
            break
    return node
