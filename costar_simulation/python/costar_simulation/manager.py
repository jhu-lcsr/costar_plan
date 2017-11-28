from __future__ import print_function

import rospy
import signal
import subprocess
import sys
import time

from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SetModelConfiguration
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse

class CostarSimulationManager(object):
    '''
    Creates and manages a gazebo simulation. Start this with:

    > rosrun costar_simulation start

    No need for a roscore to be running before hand. This will start and
    hopefully manage that roscore for you.
    '''

    model_name = "robot"
    joint_names = ["shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"]
    joint_positions = [0.30, -1.33, -1.80, -0.27, 1.50, 1.60]

    def _sampleRobotJointPositions(self):
        '''
        Note: if we are using the UR5 as the arm, we can just use these default
        position arguments and start from there.
        '''
        return self.joint_positions

    def __init__(self,launch="ur5", experiment="magnetic_assembly", seed=None,
            gui=False, case="double1", gzclient=False, *args,**kwargs):
        '''
        Define the manager and parameterize everything so that we can easily
        reset between different tasks. This also loads the appropriate task
        model and does all that other good stuff.

        Parameters:
        -----------
        launch: root name of the launch file; this determines which robot will
                be loaded and how.
        experiment: this creates the various objects that will populate the
                    world, and is used to load the appropriate task model.
        case: this is a sub-set for some of our experiments.
        gui: start RViz
        gzclient: start gazebo client
        seed: used to initialize a random seed if necessary
        '''

        # =====================
        # SANITY CHECKS: if something is a placeholder, do not use it right
        # now.
        if launch == "mobile":
            raise NotImplementedError('mobile env not yet supported')

        # Parse and set up some flags so we know how to randomize the
        # environment and how to reset things between trials
        if experiment == "magnetic_assembly":
            self.uses_gbeam_soup = True
        else:
            self.uses_gbeam_soup = False

        self.procs = []
        self.rviz = gui
        self.gui = gzclient
        self.reset_srv = None
        self.pause_srv = None
        self.unpause_srv = None
        self.experiment_file = "%s.launch"%experiment
        self.launch_file = "%s.launch"%launch

        self.seed = seed
        self.case = case
        print("=========================================================")
        print("|                 Gazebo Configuration Report            ")
        print("| -------------------------------------------------------")
        print("| launch file = %s"%self.launch_file)
        print("| experiment file = %s"%self.experiment_file)
        print("| seed = %s"%(str(self.seed)))
        print("| case = %s"%self.case)
        print("| gui = %s"%self.gui)
        print("=========================================================")

    def sleep(self, t=1.0):
        time.sleep(t)

    def reset(self):
        '''
        Reset the robot's position to its start state. Create new objects to
        manipulate based on experimental parameters.
        '''

        self.pause()
        rospy.wait_for_service("gazebo/set_model_configuration")
        configure = rospy.ServiceProxy("gazebo/set_model_configuration", SetModelConfiguration)
        configure(model_name=self.model_name,
                joint_names=self.joint_names,
                joint_positions=self.joint_positions)
        rospy.wait_for_service("gazebo/delete_model")
        if self.uses_gbeam_soup:
            delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
            delete_model("gbeam_soup")
        res = subprocess.call([
            "roslaunch",
            "costar_simulation",
            self.experiment_file,
            "experiment:=%s"%self.case])
        res = subprocess.call(["rosservice","call","publish_planning_scene"])

    def pause(self):
        if self.pause_srv is None:
            raise RuntimeError('Service Proxy not created! Is sim running?')
        self.pause_srv()

    def resume(self):
        if self.unpause_srv is None:
            raise RuntimeError('Service Proxy not created! Is sim running?')
        res = subprocess.call(["rosservice","call","publish_planning_scene"])
        self.unpause_srv()

    def reset_srv_cb(self, msg):
        self.reset()
        self.resume()
        return EmptySrvResponse()

    def run(self):
        '''
        Bring up all necessary components of the simulation. You should only
        need to do this once: after the first call to run(), you can repeatedly
        call reset() to restore the simulation to a good state.
        '''

        # ---------------------------------------------------------------------
        # Start the roscore
        roscore = subprocess.Popen(['roscore'])
        self.procs.append(roscore)
        self.sleep(1.)

        # We create a node handle so we can interact with ROS services.
        rospy.init_node('simulation_manager_node')
        
        # ---------------------------------------------------------------------
        # Start gazebo
        # This is the "launch" option -- it should dec
        gazebo = subprocess.Popen(['roslaunch',
            'costar_simulation',
            self.launch_file,
            'gui:=%s'%str(self.gui)])
        self.procs.append(gazebo)
        self.sleep(5.)

        rospy.wait_for_service('gazebo/unpause_physics')
        self.pause_srv = rospy.ServiceProxy('gazebo/pause_physics',EmptySrv)
        self.unpause_srv = rospy.ServiceProxy('gazebo/unpause_physics',EmptySrv)

        # ---------------------------------------------------------------------
        # Start rviz
        if self.rviz:
            rviz = subprocess.Popen(['roslaunch', 'costar_simulation', 'rviz.launch'])
            self.procs.append(rviz)
            self.sleep(3.)
        
        # ---------------------------------------------------------------------
        # Reset the simulation. This puts the robot into its initial state, and
        # is also responsible for creating and updating positions of any objects.
        self.reset()
        self.sleep(1.)
        self.reset()

        # ---------------------------------------------------------------------
        # Start controllers
        roscontrol = subprocess.Popen(['roslaunch', 'costar_simulation', 'controllers.launch'])
        self.procs.append(roscontrol)

        # ---------------------------------------------------------------------
        # Start moveit

        # resent the whole planning scene
        self.sleep(1.)
        subprocess.call(["rosservice","call","publish_planning_scene"])

        # ---------------------------------------------------------------------
        # Start reset service
        self.reset_srv = rospy.Service("costar_simulation/reset", EmptySrv, self.reset_srv_cb)
        self.sleep(5.)
        res = subprocess.call(["rosservice","call","publish_planning_scene"])
        self.resume()

    def shutdown(self):
        '''
        Terminate all managed processes, including spawners and CoSTAR tools.
        '''

        # Send terminate signal to all managed processes.
        for proc in reversed(self.procs):
            proc.terminate()

        # Wait and kill everything.
        self.sleep()
        try:
            for proc in reversed(self.procs):
                proc.kill()
        except Exception, e:
            pass

        # For some reason gazebo does not shut down nicely.
        self.sleep()
        subprocess.call(["pkill","gzserver"])

    def shutdownAndExitHandler(self, *args, **kwargs):
        print('You pressed Ctrl+C! Shutting down all processes.')
        self.shutdown()
        sys.exit(0)

