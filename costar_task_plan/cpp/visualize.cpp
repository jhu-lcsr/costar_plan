#include <grid/visualize.h>

#include <tf_conversions/tf_kdl.h>
#include <tf/transform_datatypes.h>

#include <trajectory_msgs/JointTrajectoryPoint.h>

#include <kdl/trajectory_composite.hpp>

using namespace KDL;
using namespace trajectory_msgs;

namespace grid {

  /*  create a pose array message from a KDL trajectory */
  geometry_msgs::PoseArray toPoseArray(Trajectory *traj, double dt, const std::string &frame) {
    geometry_msgs::PoseArray msg;
    msg.header.frame_id = frame;


    for (double t = 0; t < traj->Duration(); t += dt) {
      geometry_msgs::Pose p;
      tf::Pose tfp;
      tf::poseKDLToTF(traj->Pos(t) * KDL::Frame(KDL::Rotation::RotY(-1. * M_PI / 2)),tfp);
      tf::poseTFToMsg(tfp,p);

      msg.poses.push_back(p);
    }

    return msg;
  }
  /*  create a pose array message from a KDL trajectory */
  geometry_msgs::PoseArray toPoseArray(std::vector<Trajectory *> traj, double dt, const std::string &frame) {
    geometry_msgs::PoseArray msg;
    msg.header.frame_id = frame;

    for (unsigned int i = 0; i < traj.size(); ++i) {

      for (double t = 0; t < traj[i]->Duration(); t += dt) {
        geometry_msgs::Pose p;
        tf::Pose tfp;
        tf::poseKDLToTF(traj[i]->Pos(t) * KDL::Frame(KDL::Rotation::RotY(-1. * M_PI / 2)),tfp);
        tf::poseTFToMsg(tfp,p);

        msg.poses.push_back(p);
      }
    }

    return msg;
  }
  /*  create a pose array message from a joint trajectory */
  geometry_msgs::PoseArray toPoseArray(std::vector<trajectory_msgs::JointTrajectory> traj,
                                       const std::string &frame,
                                       RobotKinematicsPtr robot)
  {
    geometry_msgs::PoseArray msg;
    msg.header.frame_id = frame;

    for (unsigned int i = 0; i < traj.size(); ++i) {

      for (JointTrajectoryPoint &pt: traj[i].points) {
        Pose kdl_pose = robot->FkPos(pt.positions) * KDL::Frame(KDL::Rotation::RotY(-1. * M_PI / 2));
        geometry_msgs::Pose p;
        tf::Pose tfp;
        tf::poseKDLToTF(kdl_pose,tfp);
        tf::poseTFToMsg(tfp,p);

        msg.poses.push_back(p);
      }
    }

    return msg;

  }
  /*  create a pose array message from a joint trajectory */
  geometry_msgs::PoseArray toPoseArray(std::vector<trajectory_msgs::JointTrajectory> traj,
                                       const std::string &frame,
                                       RobotKinematicsPtr robot, const Pose &attached)
  {
    geometry_msgs::PoseArray msg;
    msg.header.frame_id = frame;

    for (unsigned int i = 0; i < traj.size(); ++i) {

      for (JointTrajectoryPoint &pt: traj[i].points) {
        Pose kdl_pose = robot->FkPos(pt.positions) * attached * KDL::Frame(KDL::Rotation::RotY(-1. * M_PI / 2));
        geometry_msgs::Pose p;
        tf::Pose tfp;
        tf::poseKDLToTF(kdl_pose,tfp);
        tf::poseTFToMsg(tfp,p);

        msg.poses.push_back(p);
      }
    }

    return msg;

  }
}
