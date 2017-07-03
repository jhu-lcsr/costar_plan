#!/usr/bin/env python

import unittest

from costar_task_plan.abstract import Task
from costar_task_plan.abstract import AbstractOption

import numpy as np

class PickOption(AbstractOption):
  def __init__(self, obj):
    self.obj = obj

def pick_args():
  return {
      "constructor": PickOption,
      "args": ["obj"],
      }

def pick2_args():
  return {
      "constructor": PickOption,
      "args": ["orange"],
      "remap": {"orange":"obj"},
      }

class MoveOption(AbstractOption):
  def __init__(self, obj, goal):
    self.obj = obj
    self.goal = goal

def move_args():
  return {
      "constructor": MoveOption,
      "args": ["obj","goal"],
      }

class DropOption(AbstractOption):
  def __init__(self):
    pass

def drop_args():
  return {
      "constructor": DropOption,
      "args": [],
      }

def make_template():
  task = Task()
  task.add("pick", None, pick_args())
  task.add("move", ["pick"], move_args())
  task.add("drop", ["pick","move"], drop_args())
  task.add("pick", ["drop"], None)
  return task

def make_template_test2():
  task = Task()
  task.add("pick2", None, pick2_args())
  task.add("drop", ["pick2"], drop_args())
  task.add("pick2", ["drop"], None)
  return task

def make_template_test3():
  task = Task()
  task.add("pick", None, pick_args())
  task.add("move", ["pick"], move_args())
  task.add("drop", ["move"], drop_args())
  return task

test1_res = \
"""drop() --> ["pick('obj=apple')", "pick('obj=orange')"]
move('obj=orange', 'goal=basket') --> ['drop()']
pick('obj=orange') --> ["move('obj=orange', 'goal=basket')", 'drop()']
ROOT() --> ["pick('obj=apple')", "pick('obj=orange')"]
pick('obj=apple') --> ["move('obj=apple', 'goal=basket')", 'drop()']
move('obj=apple', 'goal=basket') --> ['drop()']
"""
test1_res_b = \
"""drop() --> ["pick('obj=orange')", "pick('obj=apple')"]
move('obj=orange', 'goal=basket') --> ['drop()']
pick('obj=orange') --> ['drop()', "move('obj=orange', 'goal=basket')"]
ROOT() --> ["pick('obj=orange')", "pick('obj=apple')"]
pick('obj=apple') --> ['drop()', "move('obj=apple', 'goal=basket')"]
move('obj=apple', 'goal=basket') --> ['drop()']
"""

test2_res = \
"""drop() --> ["pick2('obj=that_one')", "pick2('obj=this_one')"]
pick2('obj=that_one') --> ['drop()']
pick2('obj=this_one') --> ['drop()']
ROOT() --> ["pick2('obj=that_one')", "pick2('obj=this_one')"]
"""
test2_res_b = \
"""drop() --> ["pick2('obj=this_one')", "pick2('obj=that_one')"]
pick2('obj=that_one') --> ['drop()']
pick2('obj=this_one') --> ['drop()']
ROOT() --> ["pick2('obj=this_one')", "pick2('obj=that_one')"]
"""

class TestTask(unittest.TestCase):

  def test1(self):
    task = make_template();
    args = {
      'obj': ['apple', 'orange'],
      'goal': ['basket'],
    }
    args = task.compile(args)
  
    self.assertEqual(len(args), 2)
    self.assertEqual(args[0]['obj'], 'apple')
    self.assertEqual(args[0]['goal'], 'basket')
    self.assertEqual(args[1]['obj'], 'orange')
    self.assertEqual(args[1]['goal'], 'basket')

    summary = task.nodeSummary()
    self.assertTrue((summary == test1_res) or summary == test1_res_b)

  def test2(self):
    task = make_template_test2();
    args = {
      'orange': ['that_one', 'this_one'],
      'goal': ['basket'],
    }
    args = task.compile(args)
    summary = task.nodeSummary()
    self.assertTrue((summary == test2_res) or summary == test2_res_b)


  def test3(self):
    task = make_template_test3();
    args = {
      'obj': ['apple', 'orange'],
      'goal': ['basket'],
    }
    args = task.compile(args)
  
    self.assertEqual(len(args), 2)
    self.assertEqual(args[0]['obj'], 'apple')
    self.assertEqual(args[0]['goal'], 'basket')
    self.assertEqual(args[1]['obj'], 'orange')
    self.assertEqual(args[1]['goal'], 'basket')

    names, seq = task.sampleSequence()
    print task.nodeSummary()
    print "Sequence = ", names
    self.assertEqual(len(seq), 3)
    self.assertEqual(len(seq), len(names))
    self.assertTrue(isinstance(seq[0], PickOption))
    self.assertTrue(isinstance(seq[1], MoveOption))
    self.assertTrue(isinstance(seq[2], DropOption))

if __name__ == '__main__':
  unittest.main()
