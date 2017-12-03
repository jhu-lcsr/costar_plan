from __future__ import print_function

import rospy
import signal
import subprocess
import sys
import time

from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse

from .experiment import GetExperiment

class CostarSimulationManager(object):
    '''
    Creates and manages a gazebo simulation. Start this with:

    > rosrun costar_simulation start

    No need for a roscore to be running before hand. This will start and
    hopefully manage that roscore for you.
    '''

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

        self.procs = []
        self.rviz = gui
        self.gui = gzclient
        self.reset_srv = None
        self.pause_srv = None
        self.unpause_srv = None
        self.publish_scene_srv = None
        self.launch_file = "%s.launch"%launch
        self.seed = seed
        self.case = case
        self.experiment = GetExperiment(experiment, case=self.case)

        print("=========================================================")
        print("|                 Gazebo Configuration Report            ")
        print("| -------------------------------------------------------")
        print("| launch file = %s"%self.launch_file)
        print("| experiment = %s"%experiment)
        print("| seed = %s"%(str(self.seed)))
        print("| case = %s"%self.case)
        print("| gui = %s"%self.gui)
        print("=========================================================")

    def sleep(self, t=1.0):
        time.sleep(t)

    def pause(self):
        if self.pause_srv is None:
            raise RuntimeError('Service Proxy not created! Is sim running?')
        self.pause_srv()

    def resume(self):
        if self.unpause_srv is None:
            raise RuntimeError('Service Proxy not created! Is sim running?')
        self.publish_scene_srv()
        self.unpause_srv()

    def reset_srv_cb(self, msg):
        '''
        Reset the robot's position to its start state. Create new objects to
        manipulate based on experimental parameters.
        '''
        self.pause()
        self.experiment.reset()
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
        self.sleep(1.)

        rospy.wait_for_service('gazebo/unpause_physics')
        rospy.wait_for_service('/publish_planning_scene')
        self.pause_srv = rospy.ServiceProxy('gazebo/pause_physics',EmptySrv)
        self.unpause_srv = rospy.ServiceProxy('gazebo/unpause_physics',EmptySrv)
        self.publish_scene_srv = rospy.ServiceProxy('/publish_planning_scene',
                                                    EmptySrv)

        # ---------------------------------------------------------------------
        # Start rviz
        if self.rviz:
            rviz = subprocess.Popen(['roslaunch', 'costar_simulation', 'rviz.launch'])
            self.procs.append(rviz)
            self.sleep(3.)
        
        # ---------------------------------------------------------------------
        # Reset the simulation. This puts the robot into its initial state, and
        # is also responsible for creating and updating positions of any objects.
        self.pause()
        self.experiment.reset()
        self.sleep(1.)
        self.experiment.reset()
        self.publish_scene_srv()

        # ---------------------------------------------------------------------
        # Start controllers
        roscontrol = subprocess.Popen(['roslaunch', 'costar_simulation', 'controllers.launch'])
        self.procs.append(roscontrol)

        # ---------------------------------------------------------------------
        # Start reset service
        self.reset_srv = rospy.Service("costar_simulation/reset", EmptySrv, self.reset_srv_cb)
        self.resume()
        self.publish_scene_srv()

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

