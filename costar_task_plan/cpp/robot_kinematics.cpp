#include <costar_task_plan/robot_kinematics.h>
#include <urdf/model.h>
#include <kdl_parser/kdl_parser.hpp>
#include <ros/ros.h>

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>

#include <trajectory_msgs/JointTrajectoryPoint.h>

using trajectory_msgs::JointTrajectory;
using trajectory_msgs::JointTrajectoryPoint;
using KDL::Trajectory;

namespace costar {


  RobotKinematics::RobotKinematics(const std::string &robot_description_param_, 
                                   const std::string &root_link_,
                                   const std::string &ee_link_)
    : robot_description_param(robot_description_param_),
    root_link(root_link_),
    ee_link(ee_link_),
    verbose(0)

  {
    ros::NodeHandle nh;
    nh.getParam(robot_description_param, robot_description);

    urdf::Model urdf_model;
    urdf_model.initString(robot_description);
    if (verbose) {
      std::cout << robot_description << std::endl;
    }
    if (!(kdl_parser::treeFromUrdfModel(urdf_model, kdl_tree))) {
      ROS_ERROR("Could not load tree!");
    }

    if (!kdl_tree.getChain(root_link, ee_link, kdl_chain)) {
      ROS_ERROR("Could not get chain from %s to %s!",root_link.c_str(),ee_link.c_str());
    } else {

      n_dof = getDegreesOfFreedom();

      joint_limits_min.resize(n_dof);
      joint_limits_max.resize(n_dof);
      hint.resize(n_dof);
      q.resize(n_dof);
      q_dot.resize(n_dof);

      // load joint limits
      // based off some of Jon's code in lcsr_controllers
      unsigned int i = 0;
      for (auto &link: kdl_chain.segments) {
        if (link.getJoint().getType() != KDL::Joint::None) {
          joint_limits_min(i) = urdf_model.joints_[link.getJoint().getName()]->limits->lower;
          joint_limits_max(i) = urdf_model.joints_[link.getJoint().getName()]->limits->upper;
          ++i;
        }
      }

      unsigned int max_iter_ik_vel = 250u;
      double tol_ik_vel = 1e-3;
      unsigned int max_iter_ik_pos = 250u;
      double tol_ik_pos = 1e-3;

      kdl_fk_solver_pos.reset(new KDL::ChainFkSolverPos_recursive(kdl_chain));
      kdl_ik_solver_vel.reset(
          new KDL::ChainIkSolverVel_wdls(
              kdl_chain,
              tol_ik_vel,
              max_iter_ik_vel
              )
          );
      kdl_ik_solver_pos.reset(
          new KDL::ChainIkSolverPos_NR_JL(
              kdl_chain,
              joint_limits_min,
              joint_limits_max,
              *kdl_fk_solver_pos,
              *kdl_ik_solver_vel,
              max_iter_ik_pos,
              tol_ik_pos
              )
          );

    }
  }

  /**
   * FkPos
   * Compute position forward kinematics
   */
  Pose RobotKinematics::FkPos(std::vector<double> &pos) {
    using KDL::JntArray;

    Pose p;
    JntArray q(n_dof);

    for (unsigned int i = 0; i < n_dof; ++i) {
      q(i) = pos[i];
    }

    kdl_fk_solver_pos->JntToCart(q,p);

    return p;
  }


  /**
   * get a list of poses this trajectory tells us to visit
   */
  std::vector<Pose> RobotKinematics::FkPos(trajectory_msgs::JointTrajectory &traj) {
    std::vector<Pose> poses(traj.points.size());
    unsigned int i = 0;
    for (auto &pt: traj.points) {
      poses[i++] = FkPos(pt.positions);
    }
    return poses;
  }


  /**
   * convert a single pose into joint positions
   */
  int RobotKinematics::IkPos (const Pose &pose, KDL::JntArray &q) {
    return kdl_ik_solver_pos->CartToJnt(hint, pose, q);
  }

  /*
   * toJointTrajectory
   * Convert trajectory into a set of poses
   * then use KDL inverse kinematics on it
   */
  bool RobotKinematics::toJointTrajectory(Trajectory *traj, JointTrajectory &jtraj, double dt) {
    std::vector<Pose> poses((unsigned int)1+floor(traj->Duration() / dt));
    std::vector<Twist> twists((unsigned int)1+floor(traj->Duration() / dt));

    unsigned int i = 0;
    for (double t = 0; t < traj->Duration(); t+=dt, ++i) {
      poses[i] = traj->Pos(t);
      twists[i] = traj->Vel(t);
    }

    return toJointTrajectory(poses, twists, jtraj, traj->Duration());
  }

  /**
   * use KDL inverse kinematics to get the trajectories back
   */
  bool RobotKinematics::toJointTrajectory(const std::vector<Pose> &poses, const std::vector<Twist> &twists, JointTrajectory &traj, double duration) {

    traj.points.resize(poses.size());

    auto pose_ptr = poses.begin();
    auto twist_ptr = twists.begin();
    KDL::JntArray q, qdot;
    for(unsigned int i = 0; i < poses.size(); ++i, ++pose_ptr) {
      int res = kdl_ik_solver_pos->CartToJnt(hint, *pose_ptr, q);
      int res2 = kdl_ik_solver_vel->CartToJnt(q, *twist_ptr, qdot);

      if (res < 0 ) return false;
      if (res2 < 0) return false;
      //else std::cout << res << std::endl;

      traj.points[i].positions.resize(n_dof);
      traj.points[i].velocities.resize(n_dof);
      //traj.points[i].accelerations.resize(n_dof);
      //traj.points[i].effort.resize(n_dof);
      for (unsigned int j = 0; j < n_dof; ++j) {
        traj.points[i].positions[j] = q(j);
        traj.points[i].velocities[j] = qdot(j);
        /*
           if (j > 0) {
           traj.points[i].accelerations[j] = 2*(qdot(j)-qdot(j-1));
           traj.points[i].effort[j] = 2*(qdot(j)-qdot(j-1));
           } else {
           traj.points[i].accelerations[j] = 2*qdot(j);
           traj.points[i].effort[j] = 2*qdot(j);
           }
           */
        //std::cout << qdot(j) << " ";
        //traj.points[i].time_from_start = ros::Duration(2 * i * (duration / poses.size()));
      }
      //std::cout << std::endl;
    }

    return true;
  }

  /**
   * just to get a jt
   */
  JointTrajectory RobotKinematics::getEmptyJointTrajectory() const {
    JointTrajectory traj;

    for (const auto &link: kdl_chain.segments) {
      traj.joint_names.push_back(link.getName());
    }

    return traj;
  }


  /**
   * get number of degrees of freedom
   */
  unsigned int RobotKinematics::getDegreesOfFreedom() const {
    return kdl_chain.getNrOfJoints();
  }

  KDL::Tree &RobotKinematics::tree() {
    return kdl_tree;
  }

  KDL::Chain &RobotKinematics::chain() {
    return kdl_chain;
  }

  /**
   * take a joint state message and use it to update KDL joints
   */
  void RobotKinematics::updateHint(const std::vector<double> &js) {
    unsigned int i = 0;
    //std::cout << "Hint: ";
    //std:: cout << hint.rows() << " " << hint.columns() << "\n";
    for(unsigned int i = 0; i < js.size(); ++i) {
      hint(i) = js.at(i);
      q[i] = js.at(i);
      //std::cout << hint(i) << " \n";
    }
    //std::cout << std::endl;
  }

    /**
     * get a hint as to the current velocity
     */
    void RobotKinematics::updateVelocityHint(const std::vector<double> &js_dot) {
      for (unsigned int i = 0; i < q_dot.size(); ++i) {
        q_dot[i] = js_dot.at(i);
      }
    }


    /**
     * get current joint positions
     */
    const std::vector<double> &RobotKinematics::getJointPos() const {
      return q;
    }

    /**
     * get current joint velocitities
     */
    const std::vector<double> &RobotKinematics::getJointVel() const {
      return q_dot;
    }
}
