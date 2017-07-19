from abstract import AbstractAgent
import pybullet as pb

class AlbertAgent(AbstractAgent):
    '''
    Really simple test agent that just generates a random set of positions to
    move to.
    '''

    name = "albert"

    def __init__(self, env, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
        self.env = env

    def fit(self, num_iter):
        a = pb.getKeyboardEvents()
        while a != ['a']:


            print a

            #self.env.step(cmd)

            if self._break:
                return
