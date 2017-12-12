
from collections import namedtuple
from learning_planning_msgs.msg import TaskInfo
from learning_planning_msgs.msg import DemonstrationInfo


NAME_STYLE_UNIQUE = 0
NAME_STYLE_SAME = 1

'''
ObjectInfo
Describe world observations of objects
'''
ObjectInfo = namedtuple(
        'ObjectInfo',
        field_names=['pose','obj_class', 'id', 'name'],
        verbose=False)

class ActionInfo(object):
    '''
    ActionInfo
    Store information about what action was being performed and by which arm. 
    This information must be provided in order to compute what type of skill
    representation we will use, and to actually create the task graph.
    '''

    ARM_LEFT = 0
    ARM_RIGHT = 1
    ARM_BOTH = 2
    ARM_POSES = [1, 1, 2]
    def __init__(self, arm, name, object_acted_on, object_in_hand, pose,
            gripper_state):
        if isinstance(arm, str):
            if arm.lower() == "left":
                arm = ARM_LEFT
            elif arm.lower() == "right":
                arm = ARM_RIGHT
            elif arm.lower() == "both":
                arm = ARM_BOTH
            else:
                raise RuntimeError('activity parse failed: arm %s not understood'%arm)
        elif not arm in [self.ARM_LEFT, self.ARM_RIGHT, self.ARM_BOTH]:
            raise RuntimeError('options are limited to LEFT, RIGHT, and BOTH.')
        self.arm = arm
        self.name = name
        self.full_name = None
        self.object_acted_on = object_acted_on
        self.object_in_hand = object_in_hand
        self.pose = pose


    def computeName(self, name_style):
        if self.name_style == NAME_STYLE_SAME:
            self.full_name = self.name
        elif self.name_style == NAME_STYLE_UNIQUE:
            self.full_name = "%s(%s)"%(self.name, self.object_acted_on)


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
        self.objects_by_class = {}
        self.classes_by_object = {}
        self.action_naming_style = action_naming_style

    def addObjectClass(self, object_class):
        self.object_classes.add(object_class)

    def addObject(self, obj, obj_class):
        self.addObjectClass(obj_class)
        if not object_class in self.objects_by_class:
            self.objects_by_class[obj_class] = set()
        self.objects_by_class[obj_class].add(obj)
        if (obj in self.classes_by_object and not obj_class == self.classes_by_object[obj]):
            raise RuntimeError("object %s has inconsistent class in data: %s vs %s"%(obj, obj_class, self.classes_by_object[obj]))
        self.classes_by_object[obj] = obj_class

    def addDemonstration(self, t, objs, actions):
        for action in actions:
            pass

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
        pass   

    actions = {}
    for demo in demonstrations:
        '''
        Look up the start and end times of the demonstrated actions
        '''

