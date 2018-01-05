
# Google Brain Grasp Dataset APIs

Author and maintainer: `Andrew Hundt <ATHundt@gmail.com>`

<img width="1511" alt="2017-12-16 surface relative transforms correct" src="https://user-images.githubusercontent.com/55744/34134058-5846b59e-e426-11e7-92d6-699883199255.png">
This version should be ready to use when generating data real training.

Plus now there is a flag to draw a circle at the location of the gripper as stored in the dataset:
![102_grasp_0_rgb_success_1](https://user-images.githubusercontent.com/55744/34133964-ccf57caa-e425-11e7-8ab1-6bba459a5408.gif)

A new feature is writing out depth image gifs:
![102_grasp_0_depth_success_1](https://user-images.githubusercontent.com/55744/34133966-d0951f28-e425-11e7-85d1-aa2706a4ba05.gif)

Image data can be resized:

![102_grasp_1_rgb_success_1](https://user-images.githubusercontent.com/55744/34430739-3adbd65c-ec36-11e7-84b5-3c3712949914.gif)

The blue circle is a visualization, not actual input, which marks the gripper stored in the dataset pose information.

Color augmentation is also available:

![102_grasp_3_rgb_success_1](https://user-images.githubusercontent.com/55744/34450881-8d017ce4-ece4-11e7-8e43-c72dcf11c89a.gif)

### How to view the vrep dataset visualization

1. copy the .ttt file and the .so file (.dylib on mac) into the `costar_google_brainrobotdata/vrep` folder.
2. Run vrep with -s file pointing to the example:
```
./vrep.sh -s ~/src/costar_ws/src/costar_plan/costar_google_brainrobotdata/vrep/kukaRemoteApiCommandServerExample.ttt
```
4. vrep should load and start the simulation
5. make sure the folder holding `vrep_grasp.py` is on your PYTHONPATH
6. cd to `~/src/costar_ws/src/costar_plan/costar_google_brainrobotdata/`, or wherever you put the repository
7. run `python2 vrep_grasp.py`