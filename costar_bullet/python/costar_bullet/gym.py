
import gym
import signal
import subprocess

from client import BulletClient
from os import path
from std_srvs.srv import Empty
from util import GetRobotInterface, GetTaskDefinition


class BulletEnv(gym.Env):
'''
Create a generic Bullet OpenAI gym environment.
'''

    def __init__(self, robot, task, *args, **kwargs):
        '''
        Create environment and set up functions to generate the appropriate
        environment.
        '''

        # TODO(cpaxton): use this to read environment parameters
        self.robot = GetRobotInterface(robot, *args, **kwargs)
        self.task = GetTaskDefinition(task, *args, **kwargs)

        # create client for this gym
        # this client abstracts out the lower-level interface PyBullet3 offers.
        self.client = BulletClient(robot, task, *args, **kwargs)

    def _step(self, action):
        '''
        Implement this method in every subclass
        Execute a single simulator step.
        '''
        self.client.step()

    def _reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def _render(self, mode="human", close=False):

        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof","-s","gzclient"]))
        else:
            self.gzclient_pid = 0

    def _close(self):

        # Kill gzclient, gzserver and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if (gzclient_count or gzserver_count or roscore_count or rosmaster_count >0):
            os.wait()

    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass

    def _seed(self):

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)  
        pass


