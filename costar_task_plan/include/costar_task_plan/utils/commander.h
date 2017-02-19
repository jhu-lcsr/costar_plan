#ifndef _GRID_COMMANDER
#define _GRID_COMMANDER

#include <ros/ros.h>

//#include <costar/trajectory.h>
#include <costar/skill.h>
#include <costar/robot_kinematics.h>

namespace costar {

  /**
   * Commander class
   * Sends messages to the robot based on a currently active skill/trajectory
   * Skills are used to determine things like whether or not we are opening the gripper
   * Trajectories are used to produce positions/velocities
   * They also need a robot kinematics object
   *
   */
  class Commander {

  public:

    Commander();

    /**
     * start executing a trajectory on the robot
     */
    int execute(Trajectory *traj);

  protected:

    ros::NodeHandle nh;

    RobotKinematicsPtr robot; // convert robot into whatever form we need
    ros::Publisher js_pub; // publish joint states
    
    SkillPtr currentSkill; // active skill


  };

}

#endif
