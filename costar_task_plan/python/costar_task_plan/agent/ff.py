from abstract import AbstractAgent

from costar_task_plan.models import MakeModel

class FeedForwardAgent(AbstractAgent):
    '''
    Simple feed forward agent. Loads everything based on model definition and
    executes in the environment.

    This does not perform any checks for what kind of model you create -- the
    only thing is that it will create the model and use model.predict() to get
    two outputs.
    
    One output is expected to be the arm position command, the other the
    gripper position command.
    '''

    name = "random"

    def __init__(self, env, *args, **kwargs):
        super(FeedForwardAgent, self).__init__(*args, **kwargs)
        self.env = env
        self.model = MakeModel(taskdef=env.task, *args, **kwargs)
        self.model.load()

    def fit(self, num_iter):

        for i in xrange(num_iter):
            print "---- Iteration %d ----"%(i+1)
            features = self.env.reset()

            while not self._break:
                cmd = self.model.predict(features)
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
        
