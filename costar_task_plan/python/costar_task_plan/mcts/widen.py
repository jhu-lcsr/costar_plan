
from abstract import AbstractWiden

'''
Implements the standard progressive widening behavior:

  if num_children < C * num_visits^alpha, widen.
  else do not.
'''


class ProgressiveWiden(AbstractWiden):

    def __init__(self, C, alpha):
        self.C = C
        self.alpha = alpha

    def __call__(self, node):
        return len(node.children) < int(self.C * node.n_visits ** self.alpha)

'''
Just another example of a function you can use.
'''


class NeverWiden(AbstractWiden):

    def __call__(self, node):
        return False
