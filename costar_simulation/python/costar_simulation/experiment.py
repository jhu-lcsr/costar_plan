
import rospy
import subprocess

from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SetModelConfiguration
from gazebo_msgs.srv import SpawnModel

from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse

class Experiment(object):
    '''
    Spawn objects
    Clean objects
    '''

    # This defines the default robot for simulating a UR5 in a particular
    # environment
    model_name = "robot"
    joint_names = ["shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"]
    joint_positions = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]

    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        raise NotImplementedError('Experiment not defined')

def GetExperiment(experiment, *args, **kwargs):
    return {
            "magnetic_assembly": MagneticAssemblyExperiment,
            "stack": StackExperiment,
            "navigation" : NavigationExperiment
            }[experiment](*args, **kwargs)

class MagneticAssemblyExperiment(Experiment):
    '''
    Magnetic assembly sim launches different blocks 
    '''

    def __init__(self, case):
        self.case = case
        self.experiment_file = "magnetic_assembly.launch"

    def reset(self):
        rospy.wait_for_service("gazebo/set_model_configuration")
        configure = rospy.ServiceProxy("gazebo/set_model_configuration", SetModelConfiguration)
        configure(model_name=self.model_name,
                joint_names=self.joint_names,
                joint_positions=self.joint_positions)
        rospy.wait_for_service("gazebo/delete_model")
        delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        delete_model("gbeam_soup")
        res = subprocess.call([
            "roslaunch",
            "costar_simulation",
            self.experiment_file,
            "experiment:=%s"%self.case])
        res = subprocess.call(["rosservice","call","publish_planning_scene"])


class StackExperiment(Experiment):
    '''
    Create a stack of blocks more or less at random
    Also probably reset the robot's joint states
    '''

    def reset(self):
        rospy.wait_for_service("gazebo/set_model_configuration")
        configure = rospy.ServiceProxy("gazebo/set_model_configuration", SetModelConfiguration)
        configure(model_name=self.model_name,
                joint_names=self.joint_names,
                joint_positions=self.joint_positions)
        pass


class NavigationExperiment(Experiment):
    '''
    Initialize a navigation experiment
    '''

    def reset(self):
        #TODO
        pass

