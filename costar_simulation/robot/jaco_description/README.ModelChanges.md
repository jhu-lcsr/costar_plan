In order to get the arm working accurately enough I had to make small changes as opposed to the documentation:

1. In transforming the DH parameters to joint angles (page 6 of the kinova DH Parameters document) I had to use 270 degrees instead of 260 for Joint 6. 
This is changed in jaco_ros/src/jaco_trajectory_action_kinova.cpp

2. For the transform of base to Joint 1 (D1 in DH document), I am using 27.05 cm, not 27.55. This is corrected in the URDF.

3. One specification for the finger transforms not given in the image *Hand.png* is the pitch of the fingers ("outwards" finger opening). Given the rotation would be aligned with the joint-to-finger transforms, from the measurements I got (in rad): thumb=-0.2658 index/pinkie=-0.2293. However, experimentally I determined that the angles are different: thumb=-0.37, index/pinkie=-0.34. 

One way to test for accuracy of the URDF model that I used is to get the finger to *just* touch the table, or parts of the jaco arm. Using RViz, one can observe whether the URDF model is doing the same.

The current model is still not perfect, even having the original measurements from Kinova did not do the job, I had to change a few parameters to mirror the real behaviour more accurately, but it is still not perfect.
