
'''
Conditions
These represent predicates that are relevant to our task. In general, we use
these to represent TERMINATION CONDITIONS for our various worlds. Call one with
a world as an argument and it will determine if the condition is no longer true
for that world.

By default, the AbstractWorld holds a list of conditions that fire if they are
violated. Each is associated with a weight.
'''

class AbstractCondition(object):

  def __init__(self):
    self.idx = 0

  def setIndex(self, idx):
    self.idx = idx

  def __call__(self, world, state, actor=None, prev_state=None):
    return self._check(world, state, actor, prev_state)

  def _check(self, world, state, actor=None, prev_state=None):
    raise Exception('condition.check() not yet implemented!')

  def name(self):
    raise Exception('condition.name() not yet implemented!')
