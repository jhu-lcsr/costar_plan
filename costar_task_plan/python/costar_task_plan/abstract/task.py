
# By Chris Paxton
# (c) 2017 The Johns Hopkins University
# See License for more details

from option import AbstractOption, NullOption
from sets import Set
from world import AbstractWorld

import copy
import numpy as np

ROOT_NAME = "ROOT"
ROOT_TAG = "ROOT()"

class TaskTemplate(object):
    def __init__(self, name, parents):
        self.name = name
        if not isinstance(parents, list):
            parents = [parents]
        self.parents = parents
        self.options = []

    def add(self, *args):
        self.options.append(args)

class Task(object):
  '''
  Model of a task as a state machine. We can specify this in any number of
  different ways: in CoSTAR, we use a Behavior Tree; PDDL or LTL can give us a
  slightly different task plan.
  '''

  def __init__(self, subtask_name=None, subtask_parents=None):
    '''
    Create the task.
    '''
    self.option_templates = {None: NullOptionTemplate()}
    self.template_connections = []

    # for subtasks
    self.subtask_name = subtask_name
    self.subtask_parents = subtask_parents

    self.compiled = False
    self.nodes = {}
    self.children = {}

    # Store integer index associated with each node
    self.indices = {}

    # Store name by integer index in case we ever want to recover that
    self.names = {}
    self.generic_names = {}

    self.conditions = []

  def mergeTask(self, task, name, inodes):
      '''
      Add subtasks and options to the current compiled task graph.

      Parameters:
      ----------
      task: the subtask to merge
      inodes: dictionary of nodes to combine

      Returns:
      --------
      inodes: updated node dictionary
      '''

      # add templates in
      for name, template in task.option_templates.items():
          self.option_templates[name] = template

      # add connections in
      for parent, child in task.template_connections:
          if parent in self.option_templates:
              self.option_templates[parent].connect(child)

      for iname, option in task.nodes.items():
          name = task.generic_names[iname]
          if iname in self.nodes:
              if name not in inodes:
                  inodes[name] = []
              if iname not in inodes[name]:
                  inodes[name].append(iname)
              for child in task.children[iname]:
                  self.children[iname].add(child)
          else:
              inodes[name] = [iname]
              self.nodes[iname] = option
              self.children[iname] = task.children[iname]

      return inodes

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
      self.option_templates[name] = OptionTemplate(
              subtask_name=self.subtask_name,
              **option_args)

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

    # Make list of args
    arg_sets = get_arg_sets(arg_dict)

    # First: connect templates (parent -> child)
    for parent, child in self.template_connections:
        if parent in self.option_templates:
            self.option_templates[parent].connect(child)

    # Possible assignments to arguments. Add children here, not outside.
    for arg_set in arg_sets:
      # List of instantiated options and subtasks, used for connecting children
      # to parents.
      # "inodes" is the list of instantiated nodes to connect. It is scoped, so
      # that for any particular set of options/substasks created, they will all
      # share the same values. This lets us create interesting hierarchical
      # structures.
      inodes = {}

      # create the nodes
      for name, template in self.option_templates.items():
        iname, option = template.instantiate(name, arg_set)
        if isinstance(option, Task):
            # this was a subtask, and must be merged into the full version of
            # the task model.
            inodes = self.mergeTask(option, name, inodes)
        else:
            self._addInstantiatedNode(name, iname, option, inodes)

      # connect nodes and their children
      for name, template in self.option_templates.items():
        if template.task is not None:
          # This subtask has been removed -- we no longer need it
          # Its associated options have been merged into the graph as a whole
          continue
        for iname in inodes[name]:
            self.generic_names[iname] = name
            for child in template.children:
              if child in inodes:
                for ichild in inodes[child]:
                  self.children[iname].add(ichild)

    if self.subtask_name == None:
        # WARNING: this is kind of terrible and might be sort of inefficient.
        # Convert to a list
        for i, (node, children) in enumerate(self.children.items()):
            children = [child for child in children]
            self.children[node] = children
            self.indices[node] = i
            self.names[i] = node

    self.compiled = True
    return arg_sets

  def _addInstantiatedNode(self, name, iname, option, inodes):
        if iname in self.nodes: # and iname is ROOT_TAG:
            return
        else:
            if name in inodes:
                inodes[name].append(iname)
            else:
                inodes[name] = [iname]
            self.nodes[iname] = option
            self.children[iname] = Set()

  def makeTree(self, world, max_depth=10):
    '''
    Make the root of a tree search. This creates the whole tree structure that
    we are going to explore, including all connections.
    '''
    depth = 0
    pass

  def index(self, name):
      '''
      Look up the index associated with a particular node ("action") so we can
      easily map to a discrete action space.
      '''
      return self.indices[name]

  def name(self, index):
      return self.names[index]

  def numActions(self):
      '''
      Indices directly indexes the array of actions so this shouldn't really
      change much.
      '''
      return len(self.names.keys())

  def nodeSummary(self):
    if not self.compiled:
      raise RuntimeError('Cannot print nodes from Task before compile() has been called!')
    summary = ''
    for name, node in self.nodes.items():
      summary += "%s --> %s\n"%(name,str(self.children[name]))
    return summary

  def sampleSequence(self):
      '''
      Sample a random sequence of options starting from the root of the task
      tree. This can be used in conjunction with many different things to get
      different executions.
      '''
      names = []
      sequence = []
      tag = ROOT_TAG
      while True:
          children = self.getChildren(tag)
          if children is None or len(children) == 0:
              break
          else:
              idx = np.random.randint(len(children))
              tag = children[idx]
              names.append(tag)
              sequence.append(self.getOption(tag))
      return names, sequence

''' =======================================================================
                        HELPERS AND INTERNAL CLASSES
    ======================================================================= '''

class OptionTemplate(object):
  '''
  Internal class that represents a single templated, non-instantiated Option.
  '''
  def __init__(self, args, constructor=None, remap=None, task=None,
          subtask_name=None, name_template="%s(%s)"):
    self.constructor = constructor
    self.subtask_name = subtask_name
    self.task = task
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
      name = ROOT_NAME
  
    if self.task is None:
        iname = self.name_template%(name,make_str(filled_args))
        option = self.constructor(**filled_args)
    else:
        option = Task(subtask_name=self.task.name)
        for args in self.task.options:
            option.add(*args)
        option.compile(arg_dict)
        iname = self.name_template%(name,
                make_str(filled_args))

    return iname, option

  def connect(self, child):
    '''
    Add a connection from this node to the child, but only if we have not
    already done so (limit 1 connection per parent-child pair).
    '''
    if child not in self.children:
        self.children.append(child)

class NullOptionTemplate(OptionTemplate):
  '''
  For the root node
  '''
  def __init__(self):
    super(NullOptionTemplate, self).__init__(
        constructor=NullOption,
        args={})

class TaskNode(object):
  '''
  Internal class that represents branches in our task search.
  '''
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

    if isinstance(vals, list):
        # loop over possible argument assignments
        for val in vals:

            # add a version where arg=val
            for prev_arg_set in prev_arg_sets:
                arg_set = copy.copy(prev_arg_set)
                arg_set[arg] = val
                arg_sets.append(arg_set)
    else:
        val = vals
        for prev_arg_set in prev_arg_sets:
            arg_set = copy.copy(prev_arg_set)
            arg_set[arg] = val
            arg_sets.append(arg_set)

  # return the set of populated assignments
  return arg_sets

def make_str(filled_args,subtask=None):
    assignment_list = []
    for arg, val in filled_args.items():
        if subtask is not None:
            assignment_list.append("%s_%s=%s")%(subtask,arg,val)
        else:
            assignment_list.append("%s=%s"%(arg,val))
    return str(assignment_list)[1:-1]
