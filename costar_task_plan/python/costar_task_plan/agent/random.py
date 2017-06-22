from abstract import AbstractAgent


class RandomAgent(AbstractAgent):
    '''
    Really simple test agent that just generates a random set of positions to
    move to.
    '''

    name = "random"

    def __init__(self, env, iter=10000, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
        self.iter = iter
        self.env = env

    def fit(self):
        for _ in xrange(self.iter):
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

        return None
        
