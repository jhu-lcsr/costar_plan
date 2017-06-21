from costar_task_plan.abstract import AbstractReward

class EuclideanReward(AbstractReward):
    '''
    Distance based purely on euclidean distance to object
    '''

    def __init__(self, goal):
        '''
        Goal is an object to approach
        '''
        self.goal = goal

    def compute(self, world):
        '''
        Reward is 0 at object.
        '''
        robot = world.actors[0]
        T_ee = pm.fromMatrix(robot.robot.forward(robot.state.arm))
        T_obj = world.getObject(self.goal).T
        print -np.linalg.norm(T_obj.p.tolist(), T_ee.p.tolist())
        return 0
