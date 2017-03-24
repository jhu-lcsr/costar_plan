#!/usr/bin/env python

try:
  from pycostar_planner import PlanningInterfaceWrapper
except ImportError, e:
  print "[COSTAR_PLANNING_INTERFACE] Could not create Boost::python bindings!"

# Hack to use roscpp stuff in python
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init

roscpp_set = False

# Wrap the C++ planning obbject in something a little more sane.
class PlanningInterface(object):

    def __init__(self,
                 robot_description="robot_description",
                 joint_states_topic="joint_states",
                 planning_scene_topic="planning_scene",
                 padding=0.,
                 num_basis_functions=5,
                 verbose=False):

        global roscpp_set
        if not roscpp_set:
            try:
                roscpp_init('costar_planning_interface',[])
            except Exception, e:
                print e

        # Create the actual interface
        self.interface = PlanningInterfaceWrapper(
                robot_description,
                joint_states_topic,
                planning_scene_topic,
                padding,
                num_basis_functions,
                verbose)

if __name__ == '__main__':
    pi = PlanningInterface()
    rospy.spin()
