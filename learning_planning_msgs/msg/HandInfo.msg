
# Description of current action, e.g. reach(orange1)
string activity

# Name of the object being acted on and of the object in the gripper
string NO_OBJECT=none
string object_acted_on
string object_in_hand

# store the status of the gripper
int8 GRIPPER_CLOSE=0
int8 GRIPPER_OPEN=1
int8 GRIPPER_FREE=2
int8 gripper_state

# pose of end effector in the world frame
geometry_msgs/Pose pose
