#!/usr/bin/env python

import unittest

from costar_task_plan.abstract import Task
from costar_task_plan.abstract import AbstractOption

class PickOption(AbstractOption):
  def __init__(self, obj):
    self.obj = obj

def pick_args():
  return {
      "constructor": PickOption,
      "args": ["obj"],
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

test1_res = """drop() --> ['pick']
move('obj=orange', 'goal=basket') --> ['drop']
pick('obj=orange') --> ['move', 'drop']
ROOT() --> ['pick']
pick('obj=apple') --> ['move', 'drop']
move('obj=apple', 'goal=basket') --> ['drop']
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
    self.assertEqual(summary, test1_res)

if __name__ == '__main__':
  unittest.main()
