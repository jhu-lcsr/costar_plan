'''
By Chris Paxton
(c) 2017 Johns Hopkins University
See license for details
'''

import sys
import signal

'''
Hold data from each rollout
'''
class Sample:
  def __init__(self, s0, a0, p0, s1, r):
    self.s0 = s0 # initial state
    self.a0 = a0 # action
    self.s1 = s1 # final state
    self.p0 = p0 # action probability
    self.r = r # reward
    self.R = 0 # return

'''
default trainer class
note that this represents an ON-POLICY trainer. Use only on-policy things to
learn models unless you're going to change things dramatically.

Things to overload:
  sample() - pull an action from the current environment
'''
class AbstractTrainer(object):

    def __init__(self, env,
        steps=10000,
        rollouts=100,
        max_rollout_length=1000,
        batch_size=1,
        discount=0.1,
        learning_rate=0.1,
        train_args={}):

        self.env = env
        self.steps = steps
        self.rollouts = rollouts
        self.batch_size = batch_size
        self.discount = discount
        self.train_args = train_args
        self.max_rollout_length = max_rollout_length
        self.learning_rate = learning_rate

        self._break = False

    '''
    Hook to add any functionality necessary to setup any models, etc.
    '''
    def compile(self, optimizer=None, *args, **kwargs):
      pass

    def _catch_sigint(self, *args, **kwargs):
      print "Caught sigint, breaking..."
      self._break = True

    def train(self, *args, **kwargs):
      self._break = False
      _catch_sigint = lambda *args, **kwargs: self._catch_sigint(*args, **kwargs)
      signal.signal(signal.SIGINT, _catch_sigint)
      print "Start training:"
      for i in xrange(self.steps):
        data, total, count = self._collect_rollouts()
        print "iter %d: collected %d samples, avg reward = %f"%(i,count,total/count)
        self._step(data, *args, **kwargs)

        if self._break:
          break

    # perform a single gradient update
    def _step(self, data):
        raise Exception('trainer._step() not implemented!')

    # draw a random action from our current model
    def sample(self, states):
        raise Exception('trainer.sample() not implemented!')

    '''
    Reset trainer state before collecting new rollouts.
    '''
    def _reset_state(self):
      pass

    '''
    Perform however many rollouts need to be executed on each time step.

    You may need to overload this function if you're doing something strange
    with your rollouts, like sampling different models (as per CEM).
    '''
    def _collect_rollouts(self):
      data = []
      total_reward = 0
      count = 0
      for i in xrange(self.rollouts):
        #print "-- ROLLOUT", i,
        done = False
        state0 = self.env.reset()
        self._reset_state()
        rollout = []
        length = 0
        while not done and not self._break:
          (action0, p0) = self.sample(state0)
          (state1, reward0, done, info) = self.env.step(action0)

          rollout.append(Sample(state0, action0, p0, state1, reward0))
          state0 = state1

          length += 1
          if length >= self.max_rollout_length:
            break

          if self._break:
            break

        if self._break:
          break

        # compute cumulative reward
        self._backup(rollout)
        for ex in rollout:
          total_reward += ex.R
          count += 1
        data.append(rollout)

      return data, total_reward, count

    '''
    Send rewards back in time. May need to be overriden for certain problems.
    This is based on the provided "discount" argument by default.
    '''
    def _backup(self, rollout):
      rollout[-1].R = rollout[-1].r
      for i in reversed(xrange(len(rollout)-1)):
        rollout[i].R = rollout[i].r + self.discount*rollout[i+1].R

    def save(self, *args, **kwargs):
      raise NotImplementedError('trainer._save() not implemented')

    def getActorModel(self):
      raise NotImplementedError('trainer.getActorModel() not implemented')

    def getCriticModel(self):
      raise NotImplementedError('trainer.getCriticModel() not implemented')

    def getAllModels(self):
      raise NotImplementedError('trainer.getAllModels() not implemented')

    '''
    Validate the results of this trainer with a simple test. This is done by 
    repeatedly calling the sample() function.
    '''
    def test(self, nb_tests=100, test_length=100, visualize=True):
      self._break = False
      _catch_sigint = lambda *args, **kwargs: self._catch_sigint(*args, **kwargs)
      signal.signal(signal.SIGINT, _catch_sigint)
      for _ in xrange(nb_tests):
        f = self.env.reset()
        terminal = False
        
        for _ in xrange(test_length):
          a, _ = self.sample(f)
          f, r, terminal, info = self.env.step(a)

          if self._break or terminal:
            break

          if visualize:
            self.env.render(mode='human')

        if self._break:
          break


