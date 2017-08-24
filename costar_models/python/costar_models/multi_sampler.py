from abstract import HierarchicalAgentBasedModel

from robot_multi_models import *
from split import *
from robot_multi_hierarchical import *

class RobotMultiSampler(RobotMultiHierarchical):

    '''
    This is the "divide and conquer"-style classifier for training a multilevel
    model. We use our supervised action labels to learn a superviser that will
    classify which action we should be performing from any particular frame,
    and then separately we learn a model of what we should be doing at each
    frame.

    This class is set up as a SUPERVISED learning problem -- for more
    interactive training we will need to add data from an appropriate agent.
    '''

    def __init__(self, taskdef, *args, **kwargs):
        '''
        Similarly to everything else -- we need a taskdef here.

        As in the other models, we call super() to parse arguments from the
        command line and set things like our optimizer and learning rate.
        '''
        super(RobotMultiHierarchical, self).__init__(taskdef, *args, **kwargs)

        self.num_frames = 1

        self.dropout_rate = 0.5
        self.img_dense_size = 1024
        self.img_col_dim = 512
        self.img_num_filters = 128
        self.combined_dense_size = 128
        self.partition_step_size = 2


