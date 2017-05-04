#!/usr/bin/env python

'''
Use this script to launch the whole simulation, not the various launch files.
'''

import signal
import subprocess
import sys
import time

class CostarSimulation(object):

    def __init__(self):
        self.procs = []

    def sleep(self):
        time.sleep(1.0)

    def run(self):

        # Start the roscore
        roscore = subprocess.Popen(['roscore'])
        self.procs.append(roscore)
        self.sleep()
        
        # Start gazebo
        gazebo = subprocess.Popen(['roslaunch', 'costar_simulation', 'ur5.launch'])
        self.procs.append(gazebo)
        self.sleep()

    def shutdown(self):
        for proc in self.procs:
            proc.terminate()

    def shutdownAndExitHandler(self):
        print('You pressed Ctrl+C! Shutting down all processes.')
        self.shutdown()
        sys.exit(0)

if __name__ == '__main__':
    sim = CostarSimulation()
    signal.signal(signal.SIGINT, sim.shutdownAndExitHandler)
    sim.run()
    signal.pause()

