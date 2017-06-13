
# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from option import AbstractOption, NullOption
from sets import Set
from world import AbstractWorld

import copy

class Task(object):
  '''
  Model of a task as a state machine. We can specify this in any number of
  different ways: in CoSTAR, we use a Behavior Tree; PDDL or LTL can give us a
  slightly different task plan.
  '''

  def __init__(self):
    '''
    Create the task.
    '''
    self.option_templates = {None: NullOptionTemplate()}
    self.template_connections = []

    self.compiled = False
    self.nodes = {}
    self.children = {}

  def add(self, name, parents, option_args):
    '''
    Note: we assume that each name uniquely identifies an action, and each
    action has a unique name.
    '''

    ignore_args = False
    if name in self.option_templates and option_args is not None:
      raise RuntimeError('Arguments for option "%s" already set!'%name)
    elif option_args is None:
      ignore_args = True

    if not ignore_args:
      self.option_templates[name] = OptionTemplate(**option_args)

    if parents is None or len(parents) == 0:
      parents = [None]
    elif not isinstance(parents, list):
      parents = [parents]

    for parent in parents:
      self.template_connections.append((parent, name))

  def getChildren(self, node):
    if node in self.children:
      return self.children[node]
    else:
      return []

  def getOption(self, node):
    if node in self.nodes:
      return self.nodes[node]
    else:
      raise RuntimeError('node %s does not exist'%node)

  def compile(self, arg_dict, unroll_depth=None):
    '''
    Instantiate this task for a particular world. This takes the task model and
    turns it into a "real" task, with appropriate options being created.

    Procedure:
     - loop over all options
     - for each option: loop over all args
     - create option with those args
    '''

    assert not self.compiled

    if isinstance(arg_dict, AbstractWorld):
        arg_dict = arg_dict.getObjects()

    # First: connect templates (parent -> child)
    for parent, child in self.template_connections:
      self.option_templates[parent].connect(child)

    # Make list of args
    arg_sets = get_arg_sets(arg_dict)

    inodes = {}

    for arg_set in arg_sets:

      # create the nodes
      for name, template in self.option_templates.items():
        iname, option = template.instantiate(name, arg_set)
        if iname in self.nodes:
          continue
        else:
          inodes[name] = iname
          self.nodes[iname] = option
          self.children[iname] = Set()

      # connect nodes and their children
      for name, template in self.option_templates.items():
        iname = inodes[name]
        for child in template.children:
            if child in inodes:
                self.children[iname].add(inodes[child])

    self.compiled = True
    return arg_sets

  def makeTree(self, world, max_depth=10):
    '''
    Make the root of a tree search. This creates the whole tree structure that
    we are going to explore, including all connections.
    '''
    depth = 0
    pass

  def nodeSummary(self):
    if not self.compiled:
      raise RuntimeError('Cannot print nodes from Task before compile() has been called!')
    summary = ''
    for name, node in self.nodes.items():
      summary += "%s --> %s\n"%(name,str(self.children[name]))
    return summary

''' =======================================================================
                        HELPERS AND INTERNAL CLASSES
    ======================================================================= '''

'''
Internal class that represents a single templated, non-instantiated Option.
'''
class OptionTemplate(object):
  def __init__(self, constructor, args, remap=None, name_template="%s(%s)"):
    self.constructor = constructor
    self.args = args
    self.remap = remap
    self.name_template = name_template
    self.children = []

  def instantiate(self, name, arg_dict):
    filled_args = {}
    for arg in self.args:
      if self.remap is not None and arg in self.remap:
        filled_arg_name = self.remap[arg]
      else:
        filled_arg_name = arg
      filled_args[filled_arg_name] = arg_dict[arg]

    if name is None:
      name = "ROOT"
  
    iname = self.name_template%(name,make_str(filled_args))
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

def get_arg_sets(arg_dict):
  
  # empty set of arguments
  arg_sets = [{}]

  # loop over all arguments
  for arg, vals in arg_dict.items():
    prev_arg_sets = arg_sets
    arg_sets = []

    # loop over possible argument assignments
    for val in vals:

      # add a version where arg=val
      for prev_arg_set in prev_arg_sets:
        arg_set = copy.copy(prev_arg_set)
        arg_set[arg] = val
        arg_sets.append(arg_set)

  # return the set of populated assignments
  return arg_sets

def make_str(filled_args):
  assignment_list = []
  for arg, val in filled_args.items():
    assignment_list.append("%s=%s"%(arg,val))
  return str(assignment_list)[1:-1]
