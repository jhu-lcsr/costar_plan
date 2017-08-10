# Costar Plan User Manual

### Basics

The easiest way to get started with costar is to type 
``` rosrun costar_bullet start --task blocks --robot ur5 --agent task  --gui ```
You will see the pybullet ExampleBrowser appear on your screen with a robotic arm that starts picking up objects.
**rosrun costar_bullet** 

#### Common errors

Remember to ``` source /opt/ros/indigo/setup.bash ``` (replace indigo with kinetic if you are using 
Ubuntu 16.04) and to ``` source devel/setup.bash ``` inside your costar_ws directory
