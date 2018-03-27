from __future__ import print_function

import numpy as np

from .util import *
from .service_caller import ServiceCaller

class StackManager(object):
    '''
    This class creates and holds the different services we need to call to
    create the stacking task and execute it.
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

    def _update(self):
        pass

    def setUpdate(self, update_fn):
        self.update = update_fn

    def reset(self):
        self.done = False
        self.current = None
        self.ok = True
        self.finished_action = False
        self.service.reset()

    def addRequest(self, parents, name, srv, req):
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
        return self.labels_list.index(label.split(':')[-1])

    def validLabel(self, label):
        return label.split(':')[-1] in self.labels

    def tick(self):
        self.finished_action = False

        # Check to make sure everything is ok
        if not self.ok:
            self.done = True

        rospy.logwarn("current = " + str(self.current))
        if self.current is not None:
            # Return status or continue
            if self.done:
                return self.ok
            elif self.service.update():
                self.done = False
                return
            elif self.current in self.children:
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
            children = self.children[self.current]
            idx = np.random.randint(len(children))
            next_action = children[idx]
            rospy.logwarn("next action = " + str(next_action))
            srv, req = self.reqs[next_action]
            self.update()
            if not self.service(srv, req):
                raise RuntimeError('could not start service: ' + next_action)
            self.current = next_action
        return self.done
