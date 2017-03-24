#ifndef PLANNING_INTERFACE_WRAPPER
#define PLANNING_INTERFACE_WRAPPER

// STL
#include <list>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

// Boost
#include <boost/thread/recursive_mutex.hpp>

// Boost Python
#ifdef GEN_PYTHON_BINDINGS
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#endif

// MoveIt!
#include <moveit/collision_detection/collision_robot.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>


// joint states
#include <sensor_msgs/JointState.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

// primitives for motion planning
#include <dmp/dmp.h>

// costar dependencies
#include <costar_task_plan/collision_map.h>

namespace costar {

  typedef trajectory_msgs::JointTrajectory Traj_t;
  typedef trajectory_msgs::JointTrajectoryPoint Traj_pt_t;

  class PlanningInterfaceWrapper {
  public:

    // Default constructor: create a planner object.
    PlanningInterfaceWrapper(const std::string &RobotDescription = std::string("robot_desciption"),
                             const std::string &JointStateTopic = std::string("joint_states"),
                             const std::string &PlanningSceneTopic = std::string("planning_scene"),
                             double padding=0.0,
                             unsigned int num_basis=5,
                             bool verbose=false);

    // Try a set of motion primitives; see if they work.
    boost::python::list pyTryPrimitives(const boost::python::list &primitives);

    // Try a single trajectory and see if it works.
    bool pyTryTrajectory(const boost::python::list &trajectory);

    // Get current joint positions.
    boost::python::list GetJointPositions() const;

    void SetVerbose(bool verbose);
    void SetK(double k);
    void SetD(double d);
    void SetTau(double tau);
    void SetGoalThreshold(double threshold);
    void SetDof(unsigned int dof);
    void SetNumBasisFunctions(unsigned int num_basis);

    // Are we allowed to collide?
    void SetCollisions(const std::string obj, bool allowed);

    // Robot object default entry
    void SetDefaultCollisions(const std::string link, bool ignore);

    void ResetCollisionMap();

    void PrintInfo() const;

  private:
    std::shared_ptr<CostarPlanner> planner;
  };
}

#endif

