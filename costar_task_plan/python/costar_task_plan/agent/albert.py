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
        #a = pb.getKeyboardEvents()
        
        for i in xrange(num_iter):
            print "---- Iteration %d ----"%(i+1)
            self.env.reset()
            print "reset reached"
            a = pb.getKeyboardEvents()

            while not self._break:
                token = 0
                print "while loop reached"
                state = self.env.world.actors[0].state
                control = SimulationRobotAction(arm_cmd=None, gripper_cmd=None)
                print "control activated"

                a = pb.getKeyboardEvents()
                print "keyboard started"
                print a
                # y opens the gripper
                if 121 in a:  
                    print "y detected"
                    # arm = 
                    token = 121
                    gripper_cmd = state.robot.gripperOpenCommand()
                    control = SimulationRobotAction(arm_cmd=None, gripper_cmd=gripper_cmd)
                    
                elif 122 in a:  
                    # arm = 
                    print "x detected"
                    token = 122
                    gripper_cmd = state.robot.gripperCloseCommand()
                    control = SimulationRobotAction(arm_cmd=None, gripper_cmd=gripper_cmd)
                    
                if 119 in a:  
                    print "w detected"
                    token = 119
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0,0,0))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                '''
                if control is not None:
                    features, reward, done, info = self.env.step(control)
                    self._addToDataset(self.env.world,
                            control,
                            features,
                            reward,
                            done,
                            i,
                            token)
                            #i,
                            #names[plan.idx])
                    if done:
                        break
                else:
                    break
                '''
            print "end statement reached"
            #self.env.step(cmd)

            if self._break:
                return
