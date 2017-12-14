
from learning_planning_msgs.msg import TaskInfo
from learning_planning_msgs.msg import DemonstrationInfo

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

