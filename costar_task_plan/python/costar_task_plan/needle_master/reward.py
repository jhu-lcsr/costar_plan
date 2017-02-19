from task_tree_search.abstract import *
from demo import *

'''
Reward function based on expert demonstrations
- load all demonstrations from filenames
- give bonus for similarity to these demonstrations
'''
class ExpertReward(AbstractReward):

  def __init__(self, filenames):
    self.filenames = filenames
  
  def evaluate(self, world):
    raise Exception('expert policy.evaluate not implemented')
    pass
