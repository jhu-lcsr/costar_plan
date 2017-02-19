
from reinforce_discrete import *
from utils import *

class ReinforceDiscreteRecurrent(ReinforceDiscrete):

  '''
  Pass everything up to the parent class
  '''
  def __init__(window_length=None,window_at_end=False,*args,**kwargs):
    super(ReinforceDiscreteRecurrent, self).__init__(*args, **kwargs)
    if not is_recurrent(actor):
      raise RuntimeError('You are using the recurrent trainer with a ' + \
          'non-recurrent model. That is a weird choice.')
    self.window_length = window_length
    self.window_at_end = window_at_end
    if self.window_length is None and not is_stateful(actor):
      raise RuntimeError('You can only use a whole sequence with a stateful rnn.')
    elif self.window_length is None:
      raise NotImplementedError('Not done!')

  def _step(self, data):
    sess = K.get_session()
    a = []
    x = []
    r = []
    for samples in data:
      length = len(samples)

      # fill out the data for this round
      aa = []
      xx = []
      rr = []
      for sample in samples:

        # Convert to one-hot
        a0 = [0]*self.nb_actions
        a0[sample.a0] = 1.

        # append data
        a.append(a0)
        x.append(sample.s0)
        r.append(sample.R)
    
    x = np.array(x)
    a = np.array(a)
    r = np.array(r)

    #uncomment to ensure shapes are what you'd expect
    #print x.shape, a.shape, r.shape

    # update the actor
    self.train_actor.run(
        session=sess,
        feed_dict={
          self.x: x,
          self.R: r,
          self.a: a,
          })

    # update the critic
    self.train_critic.run(
        session=sess,
        feed_dict={
          self.x: x,
          self.R: r,
          self.a: a,
          })
