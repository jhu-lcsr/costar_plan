from __future__ import print_function

from costar_task_plan.abstract.task_parser import TaskParser
from costar_task_plan.abstract.task_parser import ObjectInfo
from costar_task_plan.abstract.task_parser import ActionInfo

import tf_conversions.posemath as pm

import rosbag

class RosTaskParser(TaskParser):

    def __init__(self, *args, **kwargs):
        super(RosTaskParser, self).__init__(*args,**kwargs)
        self.ignore = ["NONE"]

    def fromFile(self, filename):
        bag = rosbag.Bag(filename)
        self.fromBag(bag)

    def fromBag(self, filename):
        for demo in self.demonstrations:
            t = self._getTime(demo)
            objs = self._getObjects(demo)
            print(objs)

    def _getTime(self, time):
        t = demo.header.stamp
        print(t)
        t = t.to_sec()
        return t

    def _getObjects(self, demo):
        '''
        Read in the demonstration and update object knowledge for this specific
        message instance. This mostly converts into a standard message format.

        Parameters:
        -----------
        demo: a DemonstrationInfo message
        '''
        objs = []
        for obj in demo.object:

            pose = pm.fromMsg(obj.pose)
            objs.append(
                    ObjectInfo(pose=pose,
                               obj_class=obj.object_class,
                               id=obj.id,
                               name=obj.name))
        return objs
