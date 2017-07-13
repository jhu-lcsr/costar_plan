
import keras.optimizers as optimizers

class AbstractAgentBasedModel(object):
    '''
    In CTP, models are trained based on output from a particular agent --
    possibly the null agent (which would just load a dataset). The Agent class
    will also provide the model with a way to collect data or whatever.
    '''

    def __init__(self, lr=1e-4, epochs=1000, iter=1000, batch_size=32,
            optimizer="sgd", model_descriptor="model", zdim=16, features=None,
            task=None, robot=None, *args,
            **kwargs):
        self.lr = lr
        self.iter = iter
        self.noise_dim = zdim
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model_descriptor = model_descriptor
        self.task = task
        self.features = features
        self.name = self.model_descriptor
        if self.task is not None:
            self.name += "_%s"%self.task
        if self.features is not None:
            self.name += "_%s"%self.features

    def train(self, agent, *args, **kwargs):
        raise NotImplementedError('train() takes an agent.')

    def save(self):
        '''
        Save to a filename determined by the "self.name" field.
        '''
        raise NotImplementedError('save() not supported yet.')

    def load(self):
        '''
        Load will use the current model descriptor and name to load the file
        that you are interested in, at least for now.
        '''
        raise NotImplementedError('load() not supported yet.')

    def getOptimizer(self):
        '''
        Set up a keras optimizer based on whatever settings you provided.
        '''
        optimizer = optimizers.get(self.optimizer)
        try:
            optimizer.lr = self.lr
        except Exception, e:
            print e
        return optimizer
