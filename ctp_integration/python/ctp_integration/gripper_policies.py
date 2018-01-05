
from costar_task_plan.abstract import AbstractPolicy

class CostarGripperPolicy(AbstractPolicy):

    '''
    This simple policy just looks at robot internals to send the appropriate
    "open gripper" command.
    '''

    def evaluate(self, world, state, actor):
        return SimulationRobotAction(gripper_cmd=state.robot.gripperOpenCommand())


