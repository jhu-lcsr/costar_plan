from abstract import AbstractAgent


class RandomAgent(AbstractAgent):
    '''
    Really simple test agent that just generates a random set of positions to
    move to.
    '''

    name = "random"

    def __init__(self, env, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
        self.env = env

    def fit(self, num_iter):
        for _ in xrange(num_iter):
            cmd = self.env.action_space.sample()
            print ">>>>CMD0 " + str(len (cmd))
            print ">>>>CMD0 " + str(type(cmd))


            print ">>>>CMD0 " + str(len (cmd[0]))
            print ">>>>CMD0 " + str(type(cmd[0]))
            print ">>>>CMD0 " + str(cmd[0])
            
            print ">>>>CMD1 " + str(len (cmd[1]))
            print ">>>>CMD1 " + str(type(cmd[1]))
            print ">>>>CMD1 " + str(cmd[1])

            self.env.step(cmd)

            if self._break:
                return
        
