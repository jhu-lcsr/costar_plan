
from learning_planning_msgs.msg import TaskInfo
from learning_planning_msgs.msg import DemonstrationInfo

def GenerateTaskModelFromMessages(task_info, demonstrations):
    '''
    Read through message data and use it to generate a task model.

    Parameters:
    -----------
    task_info: TaskInfo message containing data about the high-level transitions
    demonstrations: set of observations representing the world state at
                    different times
    '''
    pass


