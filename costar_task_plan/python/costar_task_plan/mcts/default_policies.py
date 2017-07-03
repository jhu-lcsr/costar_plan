from abstract import *
from node import *

# all the components we may need
from sample import *
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
                 score_c=1.0,
                 pw_C=1.0,
                 pw_alpha=0.5,
                 *args, **kwargs):

        super(DefaultMctsPolicies, self).__init__(
            score=PriorProbabilityScore(score_c),
            widen=ProgressiveWiden(pw_C, pw_alpha),
            extract=MostVisitedExtract(),
            sample=sample,
            initialize=initialize,
            *args, **kwargs)


class DefaultTaskMctsPolicies(DefaultMctsPolicies):

    '''
    Create a version of the task associated with exploring through a task
    model. This should create and initialize the right nodes associated with
    exploring the state machine that results from the task "grammar."
    '''

    def __init__(self, task, *args, **kwargs):
        super(DefaultTaskMctsPolicies, self).__init__(
            initialize=TaskModelInitialize(task),
            *args, **kwargs)
