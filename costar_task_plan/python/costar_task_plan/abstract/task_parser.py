
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
        self.base_name = name
        self.name = None
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
            min_action_length=5, # ignore anything below this length
            unknown_apply_before=4,
            configs=[],
            *args, **kwargs):
        '''
        Create a task parser. This lets you load one demonstration in at a
        time, and will parse all the necessary information to create a 
        particular object.
        '''
        self.transition_counts = {}
        self.transitions = {}
        self.object_classes = set()
        self.objects_by_class = {}
        self.classes_by_object = {}
        self.action_naming_style = action_naming_style
        self.idle_tags = []
        self.unknown_tags = []
        self.resetDemonstration()
        self.configs = configs
        self.num_arms = len(self.configs)
        self.unknown_apply_before = unknown_apply_before

    def addObjectClass(self, object_class):
        self.object_classes.add(object_class)

    def addObject(self, obj, obj_class):
        self.addObjectClass(obj_class)
        if not obj_class in self.objects_by_class:
            self.objects_by_class[obj_class] = set()
        self.objects_by_class[obj_class].add(obj)
        if (obj in self.classes_by_object and not obj_class == self.classes_by_object[obj]):
            raise RuntimeError("object %s has inconsistent class in data: %s vs %s"%(obj, obj_class, self.classes_by_object[obj]))
        self.classes_by_object[obj] = obj_class

    def _getActionName(self, action):
        if action.base_name in self.unknown_tags:
            return None
        elif self.action_naming_style == NAME_STYLE_SAME:
            return action.base_name
        elif self.action_naming_style == NAME_STYLE_UNIQUE:
            if action.arm == ActionInfo.ARM_LEFT:
                arm = "left"
            elif action.arm == ActionInfo.ARM_RIGHT:
                arm = "right"
            name = action.base_name
            if action.object_in_hand is not None:
                name += "_with_%s"%(self.classes_by_object[action.object_in_hand])
            if action.object_acted_on is not None:
                name += "_to_%s"%(self.classes_by_object[action.object_acted_on])
            return name

    def resetDemonstration(self):
        self.data = []

    def addIdle(self, *args):
        self.idle_tags += list(args)

    def addUnknown(self, *args):
        self.unknown_tags += list(args)

    def _addTransition(self, from_action, to_action):
        '''
        Helper function to populate helper data structures storing transition
        information.

        Parameters:
        ----------
        from_action: action/activity that we just executed/finished
        to_action: action/activity that we just began
        '''
        key = (from_action, to_action)
        if not from_action in self.transitions:
            self.transitions[from_action] = set()
        self.transitions[from_action].add(to_action)
        if key not in self.transition_counts:
            self.transition_counts[key] = 0
        self.transition_counts[key] += 1

    def addDemonstration(self, t, objs, actions):

        '''
        Call this to parse a single time step. This assumes you have properly
        called resetDemonstration() above.
        '''
        self.data.append((t, objs, actions))

    def processDemonstration(self):
        prev_t = [None] * self.num_arms
        prev = [None] * self.num_arms
        action_start_t = [None] * self.num_arms
        
        # Preprocess to remove "Unknown" symbols
        ACTION_IDX = 2
        for i, (t, objs, actions) in enumerate(self.data):
            for j, a in enumerate(actions):
                if a.base_name in self.unknown_tags:
                    reset_name = False
                    for k in range(self.unknown_apply_before):
                        if i+k < len(self.data):
                            next_name = self.data[i+k][ACTION_IDX][j].base_name
                            if next_name not in self.unknown_tags:
                                a.base_name = next_name
                                reset_name = True
                                break
                    if not reset_name:
                        a.base_name = prev[j]
                else:
                    prev[j] = a.base_name
                if a.base_name is None or a.base_name in self.unknown_tags:
                    print('[WARNING] was not able to preprocess unknown tag at %d'%i)

        # Reset previous tags again
        prev = [None] * self.num_arms

        # Loop over all time steps
        for i, (t, objs, actions) in enumerate(self.data):

            for obj in objs:
                self.addObject(obj.name, obj.obj_class)

            # in order - all actions specified
            for j, action in enumerate(actions):
                if prev_t[j] is not None:
                    # Trigger a sanity check to make sure we do not have any weird jumps in our file.
                    dt = abs(prev_t[j] - t)
                    if dt > 1:
                        print("WARNING: large time jump from %f to %f; did you reset between demonstrations?"%(self.prev_t[j],t))

                name = self._getActionName(action)
                if not prev[j] == name:
                    action_start_t[j] = t
                    if prev[j] is not None and name is not None:
                        self._addTransition(prev[j], name)
                if name is not None:
                    print(name, prev[j], t - action_start_t[j], action.arm)
                prev[j] = name
                prev_t[j] = t
                
        self.resetDemonstration()
