from abstract import AbstractTaskDefinition

class BlocksTaskDefinition(AbstractTaskDefinition):
    '''
    Define a simple task. The robot needs to pick up and stack blocks of
    different colors in a particular order.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Read in arguments defining how many blocks to create, where to create
        them, and the size of the blocks. Size is given as mean and covariance,
        blocks are placed at random.
        '''
        super(BlocksTaskDefinition, self).__init__(*args, **kwargs)
