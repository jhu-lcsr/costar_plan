from abstract import AbstractMctsPolicies

'''
Version of MCTS policies that uses continuous RAVE based metrics.
'''
class ContinuousMctsPolicies(AbstractMctsPolicies):
  def __init__(self,
      sample=None,
      initialize=None,
      score_c = 1.0,
      pw_C = 1.0,
      pw_alpha = 0.25,
      *args, **kwargs):

    super(ContinuousMctsPolicies, self).__init__(
      score=ContinuousRaveScore(score_c),
      widen=ProgressiveWiden(pw_C, pw_alpha),
      extract=MostVisitedExtract(),
      sample=sample,
      initialize=initialize,
      *args, **kwargs)



