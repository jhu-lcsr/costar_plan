
from dmp_policy import JointDmpPolicy, CartesianDmpPolicy

import numpy as np

from costar_task_plan.abstract import AbstractOption, AbstractCondition

# This needs to remember some bit of data from the current world. it might not
# end up being strictly necessary...
class RecordOption(AbstractOption):

  def __init__(self, observe):
    self.observe = observe
