from abstract import AbstractMctsPolicies

class ContinuousMctsPolicies(AbstractMctsPolicies):
    '''
    Version of MCTS policies that uses continuous RAVE based metrics.
    '''
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
        raise NotImplementedError('this does not yet exist')


class ContinuousSamplerTaskPolicies(AbstractMctsPolicies):
    '''
    Defines the guided version of the continous task planner used for
    optimization. It assumes continuous parameters for each action are on a
    Gaussian distribution.

    Parameters:
    -----------
    task: a Task model
    Z: dictionary mapping task model nodes to multivariate Gaussians
    '''
    def __init__(self, task, Z={},
            score_c = 1.0,
            pw_C = 1.0,
            pw_alpha = 0.5,
            unordered=True,
            *args, **kwargs):
        self.task = task
        self.Z = Z

        sample = ContinuousTaskSample(task, Z, unordered)

        super(DefaultMctsPolicies, self).__init__(
                score=PriorProbabilityScore(score_c),
                widen=ProgressiveWiden(pw_C, pw_alpha),
                extract=MostVisitedExtract(),
                sample=sample,
                initialize=None,
                *args, **kwargs)
