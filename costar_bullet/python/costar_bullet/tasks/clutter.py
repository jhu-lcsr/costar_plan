from abstract import AbstractTaskDefinition

class ClutterTaskDefinition(AbstractTaskDefinition):
    '''
    Clutter task description in general
    '''
    def __init__(self, *args, **kwargs):
        '''
        Your desription here
        '''
        super(BlocksTaskDefinition, self).__init__(*args, **kwargs)


    def init(self):
        '''
        Create random objects at random positions
        '''
        
        objs_to_add = []

        # TODO(fjonath1): choose random objects
        for i in xrange(1,randn):
            # add to objs
            pass

        for obj in objs_to_add:
            pb.loadSDF(filepath)
