#!/usr/bin/env python

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

def test1():
  task = make_template();
  args = {
    'obj': ['apple', 'orange'],
    'goal': ['basket'],
  }
  print task.compile(args)
  task.printNodes()

if __name__ == '__main__':
  test1()
