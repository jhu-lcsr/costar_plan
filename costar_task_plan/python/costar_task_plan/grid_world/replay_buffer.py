

'''
TTS class for storing data from experience replay.
'''
class ReplayBuffer(object):

  # store all the different examples: we want (s_i, a_i, r_i, s_i+1) as a tuple
  memory = []

  def __init__(self):
    pass

  def add(self, s0, a0, r0, s1):
    memory.append(s0, a0, r0, s1)

  def draw(self, num=1):
    print "not yet implemented"
    return None
