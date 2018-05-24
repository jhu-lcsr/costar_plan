from __future__ import print_function

import numpy as np

from .util import *
from .service_caller import ServiceCaller

class StackManager(object):
    '''
    A task is set up as a graph of parent action objects 
    with sets of possible child actions. To execute these actions,
    we define them to be ROS service calls.

    This class is used to walk through the graph of the things the robot 
    can do, and randomly select actions.
    Since ROS services calls are essentially remote function calls with a callback.
    this class creates and holds the different services we need to call to
    create the stacking task and execute it.

    It defines "children", a dictionary of strings/None to lists of strings.
    At each step(), it will:
      - check on its service thread and determine if its running or failed
      - if done with service call, choose a new child at random from children[current]

    Make sure you call reset() after each trial.

    Where this class organizes the actions that are going to be taken,
    the ServiceCaller class actually sends and waits to receive the messages
    for other parts of the system to execute that action.
    '''
    objs = ["red_cube", "green_cube", "blue_cube", "yellow_cube"]

    def __init__(self, *args, **kwargs):
        self.service = ServiceCaller(*args, **kwargs)
        self.detect = GetDetectObjectsService()
        self.place = GetSmartPlaceService()
        self.reset()
        self.reqs = {}
        self.children = {}
        self.labels = set()
        self.update = self._update()
        self.previous_action = None

    def _update(self):
        pass

    def setUpdate(self, update_fn):
        ''' Set the function to call after each action is completed.

          Currently this sets the function that 
          goes home and gets all the object poses after the action.
        '''
        self.update = update_fn

    def reset(self):
        self.done = False
        self.current_action = None
        self.ok = True
        self.finished_action = False
        self.service.reset()

    def addRequest(self, parents, name, srv, req):
        '''
        Define an action the robot can take, and the parent actions that
        might call it. 
        
        This includes a service call object and request message,
        so other processes in costar can execute the action.

        Parameters:
        -----------
        parents: list of strings or None indicating which actions can preceed
                 a particular action.
        name: name of this action.
        srv: service to call
        req: service call request (ROS message)
        '''
        self.reqs[name] = (srv, req)
        if not isinstance(parents, list):
            parents = [parents]
        for parent in parents:
            if parent not in self.children:
                self.children[parent] = []
            self.children[parent].append(name)
            self.labels.add(name.split(':')[-1])
            self.labels_list = list(self.labels)

    def index(self, label):
        """ Get the index of the specified action label.
        """
        return self.labels_list.index(label.split(':')[-1])

    def validLabel(self, label):
        """ Check if a label is present in the list of labels
        """
        return label.split(':')[-1] in self.labels

    def tick(self):
        """ Run one timestep during which we check if the outstanding ROS/costar actions
        are still pending, if they have completed running, or if they have encountered errors.
        """
        self.finished_action = False

        # Check to make sure everything is ok
        if not self.ok:
            self.done = True

        # print a status update for debugging purposes
        if self.previous_action is None or self.current_action != self.previous_action:
            rospy.loginfo("current = " + str(self.current_action))
            self.previous_action = self.current_action

        if self.current_action is not None:
            # Return status or continue
            if self.done:
                return self.ok
            elif self.service.update():
                self.done = False
                return
            elif self.current_action in self.children:
                # This one has a child to execute
                self.done = False
            else:
                self.done = True

        if self.service.ok:
            self.ok = True
        else:
            self.ok = False
            self.done = True
            rospy.logerr("service was not ok: " + str(self.service.result.ack))

        if not self.done:
            self.finished_action = True
            children = self.children[self.current_action]
            # choose which action to take out of the set of possible actions
            idx = np.random.randint(len(children))
            next_action = children[idx]
            rospy.logwarn("next action = " + str(next_action))
            srv, req = self.reqs[next_action]
            # Go home and use vision to update all the object poses.
            self.update()
            if not self.service(srv, req):
                raise RuntimeError('could not start service: ' + next_action)
            self.current_action = next_action
        return self.done
