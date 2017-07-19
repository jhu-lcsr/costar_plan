from abstract import AbstractAgent


class FeedForwardAgent(AbstractAgent):
    '''
    Simple feed forward agent. Loads everything based on model definition and
    executes in the environment.
    '''

    name = "random"

    def __init__(self, env, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
        self.env = env

    def fit(self, num_iter):

        for i in xrange(num_iter):
            print "---- Iteration %d ----"%(i+1)
            features = self.env.reset()

            while not self._break:
                self.env.step(cmd)
                features, reward, done, info = self.env.step(control)
                self._addToDataset(self.env.world,
                        control,
                        features,
                        reward,
                        done,
                        i,
                        names[plan.idx])
                if done:
                    break

            if self._break:
                return
        
