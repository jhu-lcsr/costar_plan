from __future__ import print_function

from costar_task_plan.abstract.task_parser import TaskParser
from costar_task_plan.abstract.task_parser import ObjectInfo
from costar_task_plan.abstract.task_parser import ActionInfo
from .lfd import LfD
from .dmp_option import DmpOption
from .dmp_policy import CartesianDmpPolicy

from learning_planning_msgs.msg import HandInfo

import tf_conversions.posemath as pm

import rosbag

class RosTaskParser(TaskParser):

    def __init__(self,
            filename=None,
            demo_topic="demonstration",
            alias_topic="alias",
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
        self.addIdle("IdleMotion")
        self.addUnknown("UnknownActivity")
        self.demo_topic = demo_topic
        self.alias_topic = alias_topic
        self.lfd = LfD(self.configs[0])
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
            elif topic == self.alias_topic:
                self.addAlias(msg.old_name, msg.new_name)
        self.processDemonstration()

    def _getArgs(self, skill_name):
        '''
        Get the args for a DMP option for creating a task graph
        
        Parameters:
        -----------
        skill_name: name of the action/skill to insert into graph
        '''

        # NOTE: hard coded for now; take the last skill. This should be either
        # a real object or the trajectory endpoint.
        obj = self.trajectory_features[skill_name][-1]

        # Create a function that will make a Cartesian skill instance.
        dmp_maker = lambda goal: DmpOption(
                goal_object=goal,
                config=self.configs[0],
                skill_name=skill_name,
                feature_model=self.lfd.skill_models[skill_name],
                kinematics=self.lfd.kdl_kin,
                traj_dist=self.lfd.getParamDistribution(skill_name),
                policy_type=CartesianDmpPolicy)

        return {
                "constructor": dmp_maker,
                "args": [obj],
                "remap": {obj: "goal"},
                }

    def _getHand(self, msg, id):
        '''
        Get the robot hand and create all appropiate fields here
        '''
        action_name = msg.activity
        obj_acted_on = msg.object_acted_on
        obj_in_gripper = msg.object_in_hand
        if (obj_acted_on == HandInfo.NO_OBJECT
                or obj_acted_on in self.ignore):
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
        objs = {}
        for obj in demo.object:
            
            obj.name = obj.name.rstrip()
            obj.object_class = obj.object_class.rstrip()
            if obj.object_class in self.ignore:
                continue

            pose = pm.fromMsg(obj.pose)
            objs[obj.name] = ObjectInfo(pose=pose,
                        obj_class=obj.object_class,
                        id=obj.id,
                        name=obj.name)
        return objs

    def train(self):
        '''
        Create the task by:
          - calling self.lfd.train() with the appropriate data
          - constructing a task model based on transitions
          - populating those transitions with the _getDmpArgs function

        Resulting task plan can be compiled and visualized as normal.
        '''
        print("===============================")
        print("-------------------------------")
        print('Observed transitions:')
        print("-------------------------------")
        for key, value in self.transitions.items():
            print (key, "has parents", list(value))
        print("-------------------------------")
        print('Observed transition counts:')
        print("-------------------------------")
        for (a, b), value in self.transition_counts.items():
            print (a, b, "seen", value, "times")
        print("-------------------------------")
        print("Number of example trajectories:")
        print("-------------------------------")
        for key, traj in self.trajectories.items():
            print("%s:"%key, len(traj), "with", self.trajectory_features[key])
        print("===============================")
        self.lfd.train(self.trajectories, self.trajectory_data, self.trajectory_features)

