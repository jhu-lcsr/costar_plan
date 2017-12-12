
from collections import named_tuple
from learning_planning_msgs.msg import TaskInfo
from learning_planning_msgs.msg import DemonstrationInfo


NAME_STYLE_UNIQUE = 0
NAME_STYLE_SAME = 1

'''
ObjectInfo
Describe world observations of objects
'''
ObjectInfo = named_tuple('pose','obj_class', 'id', 'name')

'''
ActionInfo
Store information about what action was being performed.
'''
ARM_LEFT = 0
ARM_RIGHT = 1
ARM_BOTH = 2
ARM_POSES = [1, 1, 2]
ActionInfo = named_tuple('arm', 'name', 'object_acted_on', 'object_in_hand', 'poses')


class TaskParser(object):

    def __init__(self,
            action_naming_style=NAME_STYLE_UNIQUE,
            *args, **kwargs):
        '''
        Create a task parser. This lets you load one demonstration in at a
        time, and will parse all the necessary information to create a 
        particular object.
        '''
        self.transitions = {}
        self.object_classes = set()
        self.action_naming_style = action_naming_style

    def addObjectClass(self, object_class):
        self.object_classes.add(object_class)

    def addObject(self, object, object_class):
        self.object_classes.add(ob

    def addDemonstration(self, t, objs, actions):

def GenerateTaskModelFromMessages(task_info, demonstrations, OptionType):
    '''
    Read through message data and use it to generate a task model.

    Parameters:
    -----------
    task_info: TaskInfo message containing data about the high-level transitions
    demonstrations: set of observations representing the world state at
                    different times
    OptionType: class, used to instantiate all of the different actions we
                need to represent the task.
    '''
    transitions = {}
    for transition in task_info.transition:
        

    actions = {}
    for demo in demonstrations:
        '''
        Look up the start and end times of the demonstrated actions
        '''

