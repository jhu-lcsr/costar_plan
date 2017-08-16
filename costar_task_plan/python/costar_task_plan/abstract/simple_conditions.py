from condition import AbstractCondition

class FalseCondition(AbstractCondition):
    def __call__(self,*args,**kwargs):
        return False

class TrueCondition(AbstractCondition):
    def __call__(self,*args,**kwargs):
        return True

class TimeCondition(AbstractCondition):
    '''
    TimeCondition: true until some amount of time has elapsed.
    '''
    def __init__(self,time):
        super(TimeCondition,self).__init__()
        self.time = time
    def __call__(self, world, state, actor=None, prev_state=None):
        return state.t <= self.time
    def name(self):
        return "time_condition(%d)"%self.time

class AndCondition(AbstractCondition):
    def __init__(self,*args):
        self.conditions = args
        for c in self.conditions:
            assert isinstance(c, AbstractCondition)

    def __call__(self, world, state, actor=None, prev_state=None):
        for condition in self.conditions:
            if not condition(world, state, actor, prev_state):
                return False
        else:
            return True

class OrCondition(AbstractCondition):
    def __init__(self,*args):
        self.conditions = args
        for c in self.conditions:
            assert isinstance(c, AbstractCondition)

    def __call__(self, world, state, actor=None, prev_state=None):
        for condition in self.conditions:
            if condition(world, state, actor, prev_state):
                return True
        else:
            return False

