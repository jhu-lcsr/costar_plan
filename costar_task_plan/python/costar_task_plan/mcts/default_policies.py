from abstract import *
from node import *

# all the components we may need
from score import *
from widen import *
from extract import *
from initialize import *

import numpy as np

def null_sampler(node):
  raise RuntimeError('No valid sampler provided.')
  return None

class DefaultMctsPolicies(AbstractMctsPolicies):
  def __init__(self,
      sample=None,
      initialize=None,
      score_c = 1.0,
      pw_C = 1.0,
      pw_alpha = 0.5,
      *args, **kwargs):

    super(DefaultMctsPolicies, self).__init__(
      score=PriorProbabilityScore(score_c),
      widen=ProgressiveWiden(pw_C, pw_alpha),
      extract=MostVisitedExtract(),
      sample=sample,
      initialize=initialize,
      *args, **kwargs)

class DefaultTaskMctsPolicies(DefaultMctsPolicies):
    def __init__(self, task, *args, **kwargs):
        super(DefaultTaskMctsPolicies,self).__init__(
          sample=None,
          initialize=TaskModelInitialize(task),
          *args,**kwargs)

