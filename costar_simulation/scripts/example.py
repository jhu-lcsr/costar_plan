

class CostarExample(object):
    '''
    This class contains all the information regarding a single demonstration of
    a task.
    '''

    def __init__(self,
            js_topic="joint_states",
            gripper_topic="",
            image_topic="",
            pc_topic="",
            objects={}):
        pass

    def start(self, filename):
        '''
        Create subscribers. Save data at some preset frequency.
        '''
        pass

    def finish(self, label):
        '''
        Label this data with success or failure.
        '''
        pass
