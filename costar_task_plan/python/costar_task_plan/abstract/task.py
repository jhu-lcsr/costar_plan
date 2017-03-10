
import copy

from option import AbstractOption, NullOption

'''
Model of a task as a state machine. We can specify this in any number of
different ways: in CoSTAR, we use a Behavior Tree; PDDL or LTL can give us a 
slightly different task plan.

- add an option with an argument
- 
'''
class Task(object):

  '''
  Create the task.
  '''
  def __init__(self):
    self.option_templates = {None: NullOptionTemplate()}
    self.initial_option_templates = []
    self.template_connections = []

    self.compiled = False
    self.nodes = {}
    self.initial_nodes = []
    self.T = {}

  '''
  Note: we assume that each name uniquely identifies an action, and each action
  has a unique name.
  '''
  def add(self, name, parents, option_args):

    ignore_args = False
    if name in self.option_templates and option_args is not None:
      raise RuntimeError('Arguments for option "%s" already set!'%name)
    elif option_args is None:
      ignore_args = True

    if not ignore_args:
      self.option_templates[name] = OptionTemplate(**option_args)

    if parents is None or len(parents) == 0:
      self.initial_option_templates.append(name)
      parents = [None]

    for parent in parents:
      self.template_connections.append((parent, name))

  '''
  Instantiate this task for a particular world. This takes the task model and
  turns it into a "real" task, with appropriate options being created.

  Procedure:
    - loop over all options
    - for each option: loop over all args
    - create option with those args
  '''
  def compile(self, arg_dict, unroll_depth=None):

    # First: connect templates (parent -> child)
    for parent, child in self.template_connections:
      self.option_templates[parent].connect(child)

    # make list of args
    arg_sets = get_arg_sets(arg_dict)

    for name, template in self.option_templates.items():
      for arg_set in arg_sets:
        name, option = template.instantiate(name, arg_set)

  def printNodes(self):
    if self.compiled:
      for child in self.nodes:
        pass

''' ===========================================================================
                        HELPERS AND INTERNAL CLASSES
=========================================================================== '''


'''
Internal class that represents a single templated, non-instantiated Option.
'''
class OptionTemplate(object):
  def __init__(self, constructor, args, name_template="%s(%s)"):
    self.constructor = constructor
    self.args = args
    self.name_template = name_template
    self.children = []

  def instantiate(self, name, arg_dict):
    filled_args = {}
    for arg in self.args:
      filled_args[arg] = arg_dict[arg]

    iname = self.name_template%(name,str(filled_args)[1:-1])
    option = self.constructor(**filled_args)

    return iname, option

  def connect(self, child):
    self.children.append(child)

'''
For the root node
'''
class NullOptionTemplate(OptionTemplate):
  def __init__(self):
    super(NullOptionTemplate, self).__init__(
        constructor=NullOption,
        args={})

'''
Internal class that represents branches in our task search.
'''
class TaskNode(object):
  def __init__(self, option, children=[]):
    self.option = option
    self.children = children

  def addChildren(self, children):
    for child in children:
      self.children.append(child)

def get_arg_sets(arg_dict):
  arg_sets = [{}]
  for arg, vals in arg_dict.items():
    prev_arg_sets = copy.copy(arg_sets)
    for val in vals:
      updated_arg_sets = copy.copy(prev_arg_sets)
      # add a version where arg=val
      for arg_set in prev_arg_sets:
        arg_set[arg] = val
        arg_sets.append(arg_set)

  print arg_sets
  return arg_sets
