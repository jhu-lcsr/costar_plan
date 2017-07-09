See also this wiki page][https://github.com/JenniferBuehler/jaco-arm-pkgs/wiki/Setup-Jaco-with-MoveIt]
which is the start of this article.

This README is the old documentation which will be added to the wiki soon. It is kept here for
backup purposes.

# UNDER CONSTRUCTION

The rest of this is still under development, finish with the instruction here!

The following are extracts from my old documentation.


## Integrate the sensors (optional)

*UNDER CONSTRUCTION:*: This needs to be tested again. 

If you would like MoveIt! to use the data from the sensors to build an octomap
and use it to avoid collisions, you need to load the MoveIt! sensor manager.

In the existing launch file    
  *{your-package-name}/launch/{your robot name}\_moveit\_sensor\_manager.launch.xml.*    
include     
  *$(find jaco\_description)/moveit/launch/jaco\_moveit\_sensor\_manager.launch*.    

You need to provide a .yaml file to configure the sensors, if your robot has any. You can copy    
  	*jaco\_moveit/config/sensors\_pointcloud.yaml*    
as a template, mostly only you will adapt the topic name coming from your kinect in this yaml file.    
This file has to be passed as parameter.

The resulting *{your-package-name}/launch/{your robot name}\_moveit\_sensor\_manager.launch.xml*: (replace {your-package-name}):

	<launch>
	  <include file="$(find jaco_description)/moveit/launch/jaco_moveit_sensor_manager.launch">
		<arg name="sensor_config" value="$(find {your-package-name})/config/sensors_pointcloud.yaml"/>
	  </include>
	</launch>


## Navigation launch files [optional].

*UNDER CONSTRUCTION:* This needs to be tested again.

If you have a mobile robot which you want to move around with a separate path navigator, you will need
another launch file (e.g. {your-robot}\_navigation.launch) to bring up the path control modules. This could look like this:


	<launch>
		## launch fake localization (publishes amcl_pose instead of monte carlo localization, publishing real robot pose)
		<include file="$(find navigation_actions)/launch/fake_localization.launch"/>

		## launch the path execution action server
		<include file="$(find navigation_actions)/launch/path_executor.launch">
			<arg name="path_action_topic" value="/simple_navigation/path_action" />
			<arg name="transform_path_action_topic" value="/simple_navigation/transform_path_action" />
			<arg name="cmd_vel_topic" default="/cmd_vel"/>
			<arg name="pose_topic" default="/amcl_pose"/>
			<arg name="nav_vel_min" value="0.4"/>
			<arg name="nav_vel_max" value="0.6"/>
			<arg name="nav_rot_min" value="1.5"/>
			<arg name="nav_rot_max" value="4.0"/>
			## the absolute root frame id for the world
			## (the top fixed frame for the navigating robot model)
			<arg name="nav_root_frame_id" value="/map"/>
		</include>
	</launch>

**TODO: MoveIt collision object generator package stil to be added to this repository.**

Another thing to consider is the MoveIt! collision objects. If you have sensors enabled in MoveIt!, it will keep an octomap
containing "occupied cells" in the world, meaning that for the motion planner, the robot will not be able to collide with those cubes.
The cubes are bigger than the real object though. If you want to get rid of the cube and replace it by the real object, you can
send moveIt! information about collision objects (i.e. objects the robot is allowed to collide with, e.g. to pick them up).
That removes the octomap entries and replaces them with the real object.

See an example on how to do it in the package moveit\_helper.
You can start a node which takes object information messages and publishes collision objects to MoveIt:

```
	## launch collision object generator 
	<include file="$(find moveit_helper)/launch/collision_object_generator.launch">
		## all objects which are not to be considered as collision objects (e.g. ground plane, robot itself, known objects...)
		<arg name="skip_objects" value="ground_plane" />
		## The frame in which the MoveIt! CollisionObject messages are to be expressed in
		<arg name="moveit_collision_frame_filter" value="odom"/>    
		## MoveIt! topic of service to obtain planning scene
		<arg name="moveit_get_planning_scene_topic" value="get_planning_scene"/>   
		## moveit topic where the collision objects are to be published
		<arg name="collision_object_topic" value="/collision_object"/>
		## Rate at which to publish collision messages
		<arg name="publish_collision_rate" value="10"/>
		## topic on which gazebo publishes Object.msg object information
		<arg name="gazebo_world_objects_topic" value="gazebo_world/objects"/>
		## service name on which gazebo provides Object.msg object information
		<arg name="gazebo_request_object_topic" value="gazebo_world/request_object"/>
	</include>
```

Note that this node takes *object_info_msgs::Object* messages as object information input (In future the Object.msg might
be moved to another package, as it should be independent of gazebo). You need to specify the same topic names in your gazebo launch file. 




## Run a full test of your files

**Step 1. Launch gazebo**

``roslaunch <your-package> <your-gazebo-launch-file>``    

Or if you have a more complex launch file which launches either gazebo or the real arm (e.g. jaco_ros/jaco_bringup.launch), use this one instead.

**Step 2. Launch moveit**

``roslaunch <your-robot-moveit-package-name> <robot-name>_moveit.launch``

It is important to have the argument ``load_robot:=false`` (default) if you are using gazebo.

The output should show no errors, which is a sign that the robot loaded by gazebo is
recognized and the /tf transforms are ok.

**Step 3. Launch rviz**

Now in addition, launch RViz to see if the robot takes on the same pose as in gazebo.
    
``roslaunch <your-robot-moveit-package-name> <robot-name>_rviz.launch``

Use the RViz planning interface to send a random target. Gazebo should follow the trajectory too.    

If you are using a sensor (e.g. kinect) you can add PointCloud2 to rviz and check that it arrives correctly.
Also, place an object in gazebo in front of the sensor, and observe that the octomap is shown in rviz.

*NOTE:* If you are sending planning commands via the rviz plugin, you might notice that actions can time out. You can
change the default timeout in 
  *{your-robot-moveit-package-name}/launch/trajectory_execution.launch.xml*
with the field allowed_goal_duration_margin.


**IMPORTANT:** If you change the XACRO URDF model, don't forget to re-generate the URDF (the one you loaded with th setup assistant), 
which is needed by MoveIt! To convert from xacro to URDF, use 

``rosrun xacro xacro --inorder {your-robot-xacro-filename}.urdf.xacro > {your-transformed-urdf-file>.urdf``
