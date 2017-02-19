from costar_task_search.abstract import AbstractCondition

class FalseCondition(AbstractCondition):
    def __call__(self,*args,**kwargs):
        return False

class TrueCondition(AbstractCondition):
    def __call__(self,*args,**kwargs):
        return True

'''
TimeCondition: true until some amount of time has elapsed.
'''
class TimeCondition(AbstractCondition):
  def __init__(self,time):
    super(TimeCondition,self).__init__()
    self.time = time
  def __call__(self, world, state, actor=None, prev_state=None):
    return state.t <= self.time
  def name(self):
    return "time_condition(%d)"%self.time
