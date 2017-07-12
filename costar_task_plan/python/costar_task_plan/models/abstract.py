

class AbstractAgentBasedModel(object):
    '''
    In CTP, models are trained based on output from a particular agent --
    possibly the null agent (which would just load a dataset). The Agent class
    will also provide the model with a way to collect data or whatever.
    '''

    def __init__(self, lr=1e-4, epochs=1000, iter=1000, batch_size=32,
            optimizer="sgd", *args,
            **kwargs):
        self.lr = lr
        self.iter = iter
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

    def train(self, agent, *args, **kwargs):
        raise NotImplementedError('train() takes an agent.')

