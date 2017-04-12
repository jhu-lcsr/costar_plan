
from abstract import AbstractExtract

import operator

'''
Return the most visited nodes all the way down the tree.
'''
class MostVisitedExtract(AbstractExtract):
  def __call__(self, node):
    nodes = [node]
    while not node.terminal and len(node.children) > 0:
      visits = [(i, child.n_visits) for (i, child) in enumerate(node.children)]
      max_idx, max_num_visits = max(visits, key=operator.itemgetter(1))

      node = node.children[max_idx]
      nodes.append(node)
    return nodes

'''
Return the most visited nodes all the way down the tree.

If some branches have negative expected reward, we want to ignore those -- it
does not matter how many times we visited them.
'''
class MostVisitedPositiveExtract(AbstractExtract):
  def __call__(self, node):
    nodes = [node]
    while not node.terminal and len(node.children) > 0:
      if max([child.avg_reward for child in node.children]) > 0:
        allowed = lambda child: child.avg_reward > 0
      else:
        allowed = lambda child: True
      visits = [(i, child.n_visits) for (i, child) in enumerate(node.children) if allowed(child)]
      max_idx, max_num_visits = max(visits, key=operator.itemgetter(1))

      node = node.children[max_idx]
      nodes.append(node)
    return nodes

'''
Extract children with the highest average reward.
'''
class HighestAverageRewardExtract(AbstractExtract):
  def __call__(self, node):
    nodes = [node]
    while not node.terminal and len(node.children) > 0:
      reward = [(i, child.avg_reward) for (i, child) in enumerate(node.children)]

      if len(reward) == 0:
          break

      max_idx, max_rewar = max(reward, key=operator.itemgetter(1))

      node = node.children[max_idx]
      nodes.append(node)
    return nodes


