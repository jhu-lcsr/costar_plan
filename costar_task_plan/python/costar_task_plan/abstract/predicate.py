
'''
New and improved successor to the Condition.

Each one has an ORDER and DEPENDENCIES.
'''
class Predicate(AbstractCondition):

  def __init__(self,deps):
    self.deps = deps
    pass

  
