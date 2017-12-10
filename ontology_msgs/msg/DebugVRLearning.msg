std_msgs/Header header

# Contain information on objects in scene
ObjectData[] object

#contain the information about the left hand
geometry_msgs/Vector3 velocity_L
string activity_L
string object_acted_on_L
string object_in_hand_L


#contain the information about the right hand
geometry_msgs/Vector3 velocity_R
string activity_R
string object_acted_on_R
string object_in_hand_R

