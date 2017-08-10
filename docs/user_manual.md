# Costar Plan User Manual

## Basics

The easiest way to get started with costar is to type 
``` rosrun costar_bullet start --task blocks --robot ur5 --agent task  --gui --show_images```
You will see the pybullet ExampleBrowser appear on your screen with a robotic arm that starts picking up objects.

**rosrun costar_bullet start** simply runs the **start** node from within the costar_bullet package. If you wish to
modify the setup of costar_plan then you can take a look 
[here](https://github.com/cpaxton/costar_plan/tree/master/costar_bullet/scripts), 
in particular at the **gui_start** program. 

The ```--task``` tag refers to the task the arm is trying to complete. The task
in this case is called **blocks** and can be found [here](https://github.com/cpaxton/costar_plan/blob/c375e723bcdf65634253b6954076d0a41070ba71/costar_task_plan/python/costar_task_plan/simulation/tasks/blocks.py). While the
blocks task is the one that is at the furthest stage of development, other tasks like **obstructions** and **obstacles** are worth
taking a look at as well. 

The **ur5** in the ```--robot ``` tag dictates the type of arm that will be used in the simulation. Several arms are currently available for development, however the ur5 arm is the only one that is officially supported at this time. Adding the necessary components and linking can be a bit complicated for integrating costar_plan with a different robotic arm and thus will be added to a separate guide shortly.

Finally the ```--gui``` tag just specifies that the gui window should appear on your screen. The gui window is most userful for debugging since with more complex scenarios it can be rather computationally intensive. Likewise the ``` show_images ``` tag - which displays the RGB, depth, and mask images for the arm - should also be used for debugging.

### Common errors

Remember to ``` source /opt/ros/indigo/setup.bash ``` (replace indigo with kinetic if you are using 
Ubuntu 16.04) and to ``` source devel/setup.bash ``` inside your costar_ws directory. If you are experiencing errors with either
glewXInit or GL rendering, try updating your drivers if you are using an NVIDIA graphics card or using **gui_start** instead of **start**.


