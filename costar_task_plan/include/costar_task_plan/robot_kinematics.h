#ifndef _GRID_ROBOT_KINEMATICS
#define _GRID_ROBOT_KINEMATICS

//#include <grid/features.h>
//
#include <kdl/frames.hpp>
#include <kdl/trajectory.hpp>

#include <kdl/chain.hpp>
#include <kdl/tree.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainiksolver.hpp>

#include <sensor_msgs/JointState.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <memory>

namespace grid {

  // grid (for robots) is going to be based on KDL
  typedef KDL::Frame Pose;
  typedef KDL::Twist Twist;

  /** 
   * Robot kinematics object.
   * For now it's an Orocos thing.
   * Pass one of these to any particular entity and it will be able to provide the inverse kinematics solving
   * /other stuff necessary to run things behind the scenes.
   *
   * TODO: make this OrocosRobotKinematics, have it inherit from RobotKinematics
   * This way we can have other things like VREP support
   */
  class RobotKinematics {
  public:

    RobotKinematics(const std::string &robot_description_param, const std::string &root_link, const std::string &ee_link);

    KDL::Tree &tree();
    KDL::Chain &chain();

    /**
     * FkPos
     * Compute position forward kinematics
     */
    Pose FkPos(std::vector<double> &pos);

    /**
     * get number of degrees of freedom
     */
    unsigned int getDegreesOfFreedom() const;

    /*
     * toJointTrajectory
     * Convert trajectory into a set of poses
     * then use KDL inverse kinematics on it
     */
    bool toJointTrajectory(KDL::Trajectory *traj,
                           trajectory_msgs::JointTrajectory &jtraj,
                           double dt=0.05);

    /**
     * use KDL inverse kinematics to get the trajectories back
     */
    bool toJointTrajectory(const std::vector<Pose> &poses,
                           const std::vector<Twist> &twists, 
                           trajectory_msgs::JointTrajectory &jtraj,
                           double duration=1);

    /**
     * convert a single pose into joint positions
     */
    int IkPos (const Pose &pose, KDL::JntArray &q);

    /**
     * take a joint state message and use it to update KDL joints
     */
    void updateHint(const std::vector<double> &js);

    /**
     * get a hint as to the current velocity
     */
    void updateVelocityHint(const std::vector<double> &js_dot);


    /**
     * get current joint positions
     */
    const std::vector<double> &getJointPos() const;

    /**
     * get current joint velocitities
     */
    const std::vector<double> &getJointVel() const;

    /**
     * just to get a jt
     */
    trajectory_msgs::JointTrajectory getEmptyJointTrajectory() const;

    /**
     * get a list of poses this trajectory tells us to visit
     */
    std::vector<Pose> FkPos(trajectory_msgs::JointTrajectory &traj);

  protected:
    std::string robot_description_param;
    std::string robot_description;
    std::string ee_link;
    std::string root_link;
    KDL::Chain kdl_chain;
    KDL::Tree kdl_tree;
    std::shared_ptr<KDL::ChainFkSolverPos> kdl_fk_solver_pos;
    std::shared_ptr<KDL::ChainIkSolverVel> kdl_ik_solver_vel;
    std::shared_ptr<KDL::ChainIkSolverPos> kdl_ik_solver_pos;
    unsigned int n_dof;
    int verbose;

    std::vector<double> q;
    std::vector<double> q_dot;

    KDL::JntArray joint_limits_min;
    KDL::JntArray joint_limits_max;

    KDL::JntArray hint;

  };

  typedef std::shared_ptr<RobotKinematics> RobotKinematicsPtr;
}

#endif
