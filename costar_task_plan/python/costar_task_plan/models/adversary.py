
import numpy as np

from costar_task_plan.abstract.params import Params

'''
Choose the opt we are going to use for our next trial.
'''
class AbstractAdversary(object):


  def __init__(self, opt, gamma=0.99):
    self.param_shape = opt.getParamShape()
    self.idx = 0
    self.R = 0
    self.gamma = gamma
    self.option = opt

  def update(self, reward):
    self.R += reward

  '''
  Helpful function: get a new world to play around in
  '''
  def makeWorld(self,*args,**kwargs):
    params = self.sample()
    return self.option.makeWorld(params,*args,**kwargs)

  '''
  Update and finish
  '''
  def finish(self):
    self.R = 0

  '''
  Returns: params
  '''
  def sample(self):
    raise NotImplementedError('adversary.sample() must be implemented!')

'''
Randomly generate a world configuration. No prior.
'''
class RandomAdversary(AbstractAdversary):

  def __init__(self, opt, *args, **kwargs):
    super(RandomAdversary, self).__init__(opt)

  def sample(self):
    return Params(self.param_shape)

  def update(self,*args,**kwargs):
    pass

'''
Store a large library of possible worlds.
Either choose one totally at random, or choose one weighted based on how badly
we've done on it in the past.
'''
class HostileAdversary(AbstractAdversary):
  def __init__(self, opt,
      burn_in=100,
      cache_size=1000,
      p_hostile=0.25,
      *args, **kwargs):
    super(HostileAdversary, self).__init__(opt)
    self.cache = [0]*cache_size
    self.sum = 0
    self.calls = 0
    self.burn_in = burn_in
    self.p_hostile = p_hostile
    self.cache_size = cache_size

    for i in xrange(cache_size):
      self.cache[i] = [Params(self.param_shape), 0]

  def sample(self):
    self.calls += 1
    #print self.calls, self.burn_in, self.p_hostile
    if self.calls > self.burn_in and np.random.random() < self.p_hostile:
      # weight based on last known reward
      idx = 0
      p = np.array([-p for (param, p) in self.cache])
      #print ""
      #print "-------------------------------------------------"
      #print "INVERSE SCORES OF ALL SCENARIOS"
      #print p
      minp = np.min(p)
      p -= minp
      p /= np.sum(p)
      #print p
      idx = np.random.choice(self.cache_size,p=p)

    else:
      idx = np.random.randint(0,self.cache_size)
    
    self.idx = idx
    return self.cache[idx][0]

  '''
  Update and finish
  '''
  def finish(self):
    self.cache[self.idx][1] = self.R
    self.cache[self.idx][0].reset()
    self.R = 0
      

def getAdversary(adv, opt, *args, **kwargs):

  if adv == "random":
    return RandomAdversary(opt, *args, **kwargs)
  elif adv == "hostile":
    return HostileAdversary(opt, *args, **kwargs)

def getAvailableAdversaries():
  return ["random", "hostile"]
