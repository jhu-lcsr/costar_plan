from __future__ import print_function

from costar_task_plan.abstract.task_parser import TaskParser
from costar_task_plan.abstract.task_parser import ObjectInfo
from costar_task_plan.abstract.task_parser import ActionInfo

from learning_planning_msgs.msg import HandInfo

import tf_conversions.posemath as pm

import rosbag

class RosTaskParser(TaskParser):

    def __init__(self,
            filename=None,
            demo_topic="demonstration",
            *args, **kwargs):
        '''
        Create the ROS version of the task parser -- represents a given task
        as a set of DMPs of various sorts.

        There are a couple frames that we ignore by default: the "NONE" object,
        the "surveillance camera" that represents a view of the scene, and
        possibly others in the future.

        Parameters:
        -----------
        filename: name of the bag file to load when parsing messages
        '''
        super(RosTaskParser, self).__init__(*args,**kwargs)
        self.ignore = ["NONE","none","surveillance_camera"]
        self.ignore_actions = ["UnknownActivity","IdleMotion"]
        self.demo_topic = demo_topic
        if filename is not None:
            self.fromFile(filename)

    def fromFile(self, filename):
        bag = rosbag.Bag(filename, 'r')
        self.fromBag(bag)

    def fromBag(self, bag):

        # call whenever adding a new rosbag or data source for a particular
        # trial.
        self.resetDemonstration()
        for topic, msg, _ in bag:
            # We do not trust the timestamps associated with the bag since
            # these may be written separately from when the data was actually
            # collected.
            if topic == self.demo_topic:
                # Demo topic: read object information and add
                t = self._getTime(msg)
                objs = self._getObjects(msg)
                left = self._getHand(msg.left, ActionInfo.ARM_LEFT)
                right = self._getHand(msg.right, ActionInfo.ARM_RIGHT)
                self.addDemonstration(t, objs, [left, right])
        print(self.transitions)
        print(self.transition_counts)


    def _getHand(self, msg, id):
        '''
        Get the robot hand and create all appropiate fields here
        '''
        action_name = msg.activity
        obj_acted_on = msg.object_acted_on
        obj_in_gripper = msg.object_in_hand
        if (obj_acted_on == HandInfo.NO_OBJECT
            or obj_in_gripper in self.ignore):
            obj_acted_on = None
        if (obj_in_gripper == HandInfo.NO_OBJECT
            or obj_in_gripper in self.ignore):
            obj_in_gripper = None
        pose = pm.fromMsg(msg.pose)
        gripper_state = msg.gripper_state
        return ActionInfo(id, action_name, obj_acted_on, obj_in_gripper, pose,
                gripper_state)

    def _getTime(self, demo):
        t = demo.header.stamp
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

            if obj.object_class in self.ignore:
                continue

            pose = pm.fromMsg(obj.pose)
            objs.append(
                    ObjectInfo(pose=pose,
                               obj_class=obj.object_class,
                               id=obj.id,
                               name=obj.name))
        return objs
