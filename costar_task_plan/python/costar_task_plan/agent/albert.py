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
            a = pb.getKeyboardEvents()

            while not self._break:
                token = 0
                state = self.env.world.actors[0].state
                control = SimulationRobotAction(arm_cmd=None, gripper_cmd=None)

                a = pb.getKeyboardEvents()
                if a != {}: print a
                # y opens the gripper
                if 121 in a:  
                    print "y detected"
                    # arm = 
                    token = 121
                    gripper_cmd = state.robot.gripperOpenCommand()
                    control = SimulationRobotAction(arm_cmd=None, gripper_cmd=gripper_cmd)
                    
                elif 122 in a:  
                    # arm = 
                    print "z detected"
                    token = 122
                    gripper_cmd = state.robot.gripperCloseCommand()
                    control = SimulationRobotAction(arm_cmd=None, gripper_cmd=gripper_cmd)
                    
       #############################Forwards##########################
    
                if 119 in a:  
                    print "w detected"
                    token = 119
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0.1,0,0))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                
                if 97 in a:  
                    print "a detected"
                    token = 97
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0,0.1,0))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                
                if 114 in a:  
                    print "r detected"
                    token = 114
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0,0,0.1))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                
                if 113 in a:  
                    print "q detected"
                    token = 113
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0.1,0.1,0.1))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
      
    ############################Backwards#########################
    
                if 115 in a:  
                    print "s detected"
                    token = 115
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(-0.1,0,0))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                
                
                if 100 in a:  
                    print "d detected"
                    token = 100
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0,-0.1,0))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                
                if 102 in a:  
                    print "f detected"
                    token = 102
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0,0,-0.1))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                
                if 101 in a:  
                    print "e detected"
                    token = 101
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(-0.1,-0.1,-0.1))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd):
                    
                if 104 in a:
                    print("---------Help requested--------------")
                    print(" ------------------------------------");
                    print("| Jaco Keyboard Teleop Help          |");
                    print("|------------------------------------|*");
                    print("| Current Mode: Arm Control          |*");
                    print("|------------------------------------|*");
                    print("| w/s : forward/backward translation |*");
                    print("| a/d : left/right translation       |*");
                    print("| r/f : up/down translation          |*");
                    print("| q/e : roll                         |*");
                    print("| up/down : pitch                    |*");
                    print("| left/right : yaw                   |*");
                    print("| 2 : switch to Finger Control       |*");
                    print(" ------------------------------------**");
                    print("  *************************************");
                
                if control is not None:
                    features, reward, done, info = self.env.step(control)
                    '''
                    self._addToDataset(self.env.world,
                            control,
                            features,
                            reward,
                            done,
                            i,
                            token)
                            #i,
                            #names[plan.idx])
                    '''
                    if done:
                        break
                else:
                    break
            
            print "end statement reached"
            #self.env.step(cmd)

            if self._break:
                return
