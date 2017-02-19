#ifndef _GRID_DEBUG
#define _GRID_DEBUG

#include <grid/features.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>

#include <grid/robot_kinematics.h>

#include <vector>

#include <trajectory_msgs/JointTrajectory.h>

namespace grid {

  /*  create a pose array message from a KDL trajectory */
  geometry_msgs::PoseArray toPoseArray(Trajectory *traj,
                                       double dt,
                                       const std::string &frame);

  /*  create a pose array message from a KDL trajectory */
  geometry_msgs::PoseArray toPoseArray(std::vector<Trajectory *> traj,
                                       double dt,
                                       const std::string &frame);

  /*  create a pose array message from a joint trajectory */
  geometry_msgs::PoseArray toPoseArray(std::vector<trajectory_msgs::JointTrajectory> traj,
                                       const std::string &frame,
                                       RobotKinematicsPtr robot);
  /*  create a pose array message from a joint trajectory */
  geometry_msgs::PoseArray toPoseArray(std::vector<trajectory_msgs::JointTrajectory> traj,
                                       const std::string &frame,
                                       RobotKinematicsPtr robot,
                                       const Pose &attached);
}

#endif
