
from option import *

'''
Model of a task as a state machine. We can specify this in any number of
different ways: in CoSTAR, we use a Behavior Tree; PDDL or LTL can give us a 
slightly different task plan.

- add an option with an argument
- 
'''
class Task(object):

  '''
  Internal class that represents a single templated, non-instantiated Option.
  '''
  class OptionTemplate(object):
    def __init__(self, constructor, name_template, args):
      self.constructor = constructor
      self.args = args
      self.name_template = name_template

  '''
  Create the task.
  '''
  def __init__(self):
    self.option_templates = {}

    self.compiled = False
    self.options = {}
    self.initial_options = []
    self.T = []

  '''
  Note: we assume that each action is identical.
  '''
  def add(self, name, parents, option_args):

    if name in self.options and option_args is not None:
      raise RuntimeError('Arguments for option "%s" already set!'%name)

    if parents is None or len(parents) == 0:
      self.initial_options.append(name)

    

  '''
  Instantiate this task for a particular world. This takes the task model and
  turns it into a "real" task, with appropriate options being created.

  Procedure:
    - loop over all options
    - for each option: loop over all args
    - create option with those args
  '''
  def compile(self, arg_dict, tree=False):
    pass

