#ifndef GRID_PLANNER
#define GRID_PLANNER

/* Grounded Robot Instruction from Demonstrations
 * ---
 *  This particular file contains the "fast" version of the C++ GRID planner.
 *  It depends on a modified DMP library, so that we can quickly search for DMP trajectories that will work.
 */

// General ROS dependencies
#include <ros/ros.h>

// STL
#include <list>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <math.h>

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

  //typedef std::vector< std::vector <double> > Traj_t;
  typedef trajectory_msgs::JointTrajectory Traj_t;
  typedef trajectory_msgs::JointTrajectoryPoint Traj_pt_t;

  /**
   * CostarPlanner
   * This class defines a single planner interface.
   * It is associated with a robot model and listens to a planning scene.
   * To plan, we associate an object key ("link", for example) with a particular object in the world via TF frame.
   *
   * So, a query would go:
   *  CostarPlanner gp();
   *  ...
   *  world = std::unordered_map<std::string,std::string>();
   *  world["link"] = "gbeam_link_1/gbeam_link";
   *  gp.plan("approach","grasp",world);
   *
   *  We instantiate a CostarPlanner by providing a set of labeled demonstrations for each "state".
   *  We plan from state A at time 0 to state B at time 0.
   *
   *  Each state is associated with a single (FOR NOW) Gaussian representing features.
   *
   *  --- ON PLANNING ---
   *  The key step for planning is just repeatedly sampling new primitives from some distribution.
   *  For the sake of saving implementation time, the way we do this is to sample from our distibution in PYTHON and call this code.
   *  There is a python-targeted version of the TryPrimitives helper function we can use.
   *  Then the responsibilities of this code are to:
   *  - maintain an accurate PlanningScene
   *  - maintain a robot model (with joint states, etc.)
   *  - call DMP code to generate trajectories
   */
  class CostarPlanner {

  public:

    // Constructor
    CostarPlanner(const std::string &RobotDescription = std::string("robot_desciption"),
                const std::string &JointStateTopic = std::string("joint_states"),
                const std::string &PlanningSceneTopic = std::string("planning_scene"),
                double padding=0.0,
                unsigned int num_basis=5,
                bool verbose=false);

    // Destructor
    ~CostarPlanner();

    static const std::string TIME;
    static const std::string GRIPPER; // fixed to BHand for now!
    static const std::string PS_TOPIC;

    /* instantiate a planning request with the given objects */
    bool Plan(const std::string &action1,
              const std::string &action2,
              const std::unordered_map<std::string, std::string> &object_mapping);

    /* add an object to the action here */
    bool AddObject(const std::string &object_name);

    /* add an action */
    bool AddAction(const std::string &action_name);

    /* try a set of motion primitives; see if they work. */
    Traj_t TryPrimitives(std::vector<double> primitives);

    /* try a single trajectory and see if it works. */
    bool TryTrajectory(const std::vector <std::vector<double> > &traj);

    /*
     * try a single trajectory and see if it works.
     * this is the joint trajectory version (so we can use a consistent message type)
     * */
    bool TryTrajectory(const Traj_t &traj, unsigned int step = 1);

    /* update planning scene topic */
    void  SetPlanningSceneTopic(const std::string &topic);
  
    /* configure degrees of freedom */
    void SetDof(const unsigned int dof);
      
    /* configure number of basis functions */
    void SetNumBasisFunctions(const unsigned int num);

    void SetK(const double k_gain);
    void SetD(const double d_gain);
    void SetTau(const double tau);
    void SetGoalThreshold(const double threshold);

    /* Are we allowed to collide? */
    void SetCollisions(const std::string obj, bool allowed);

    /* Robot object default entry */
    void SetDefaultCollisions(const std::string link, bool ignore);

    void SetVerbose(const bool verbose);

    void PrintInfo() const;

    const std::vector<double> &currentPos() const;

    const std::vector<double> &currentVel() const;

    /* reset all entries in the collision map */
    void ResetCollisionMap();

  private:
    CollisionMap cm;

    //std::unordered_map<std::string, std::string> object_lookup;
    
    robot_model_loader::RobotModelLoaderPtr robot_model_loader;
    robot_model::RobotModelPtr model;
    //planning_scene_monitor::PlanningSceneMonitorPtr monitor;

    unsigned int dof;
    unsigned int num_basis;
    double k_gain;
    double d_gain;
    double tau;
    double threshold;

    //boost::shared_ptr<robot_state::RobotState> search_state;
    //boost::shared_ptr<robot_state::RobotState> state;
    //boost::shared_ptr<planning_scene::PlanningScene> scene;
    //robot_state::RobotState *search_state;
    //robot_state::RobotState *state;
    planning_scene::PlanningScene *scene;

    std::vector<double> goal;
    std::vector<double> goal_threshold;
    std::vector<double> x0;
    std::vector<double> x0_dot;

    std::vector<std::string> joint_names;
    std::vector<std::string> entry_names;

    bool verbose;

    mutable boost::recursive_mutex ps_mutex;
    mutable boost::recursive_mutex js_mutex;

    ros::NodeHandle nh;

    ros::Subscriber js_sub;
    ros::Subscriber ps_sub;

    /* keep robot joints up to date */
    void JointStateCallback(const sensor_msgs::JointState::ConstPtr &msg);

    /* keep scene up to date */
    void PlanningSceneCallback(const moveit_msgs::PlanningScene::ConstPtr &msg);

    friend class PlanningInterfaceWrapper;
  };
}

#endif
