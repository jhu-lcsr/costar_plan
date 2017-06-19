

class Grammar(object):
  '''
  Alternate task definition strategy
  '''

  RULE = "rule"
  CONDITION = "test"
  OPTION = "primitive"


  def __init__(self):
    # these are primitives/tokens
    self.options = {}
    # these are "phrases"
    self.rules = {}
    # these are tests at action completion
    self.conditions = {}

  def _check(self, name, rule_type)
    '''
    Rules can be repeated, but conditions and options cannot.
    '''
    if name in self.options or name in self.conditions:
      raise RuntimeError('%s already defined'%name)
    if name in self.rules and rule_type is not self.RULE:
      raise RuntimeError('cannot redefine rule as %s (name=%s)'%(rule_type, name))

  def definePrimitive(self, name, option_maker):
    self._check(name, OPTION)
    self.options[name] = option_maker

  def defineTest(self, name, condition_maker):  
    self._check(name, OPTION)
    self.conditions[name] = condiiton_maker

  def defineRule(self, name, rule):
    if name in self.options or name in self.rules:
      raise RuntimeError('%s already defined'%name)
    self.rules[name] = rule

  def compile(self, entry=None, max_depth=5):
    '''
    Start at root. Compile any rules for the root.
    '''

    # This makes sure we don't exceed our max depth for any particular rule.
    depth = {}
    pass
