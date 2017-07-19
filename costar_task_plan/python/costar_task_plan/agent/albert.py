from abstract import AbstractAgent
import pybullet as pb
import PyKDL as kdl
from costar_task_plan.simulation.world import *

class AlbertAgent(AbstractAgent):
    '''
    Really simple test agent that just generates a random set of positions to
    move to.
    '''

    name = "albert"

    def __init__(self, env, *args, **kwargs):
        super(AlbertAgent, self).__init__(*args, **kwargs)
        self.env = env

    def fit(self, num_iter):
        a = pb.getKeyboardEvents()

        for i in xrange(num_iter):
            print "---- Iteration %d ----"%(i+1)
            self.env.reset()

            while not self._break:
                state = self.env.world.actors[0].state
                control = SimulationRobotAction(arm_cmd=None, gripper_cmd=None)

                a = pb.getKeyboardEvents()
                
                # y opens the gripper
                if 121 in a:  
                    # arm = 
                    gripper_cmd = state.robot.gripperOpenCommand()
                    control = SimulationRobotAction(arm_cmd=None, gripper_cmd=gripper_cmd)
                    
                elif 122 in a:  
                    # arm = 
                    gripper_cmd = state.robot.gripperCloseCommand()
                    control = SimulationRobotAction(arm_cmd=None, gripper_cmd=gripper_cmd)
                    
                if 119 in a:  
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0,0,0))
                    arm = origin * move
                    gripper_cmd = None
                    invarm = state.robot.ik(T_step, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)

                if control is not None:
                    features, reward, done, info = self.env.step(control)
                    self._addToDataset(self.env.world,
                            control,
                            features,
                            reward,
                            done,
                            i,
                            names = "demo")
                            #i,
                            #names[plan.idx])
                    if done:
                        break
                else:
                    break

            #self.env.step(cmd)

            if self._break:
                return
