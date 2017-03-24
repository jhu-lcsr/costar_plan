#include <costar_task_plan/costar_planner.h>
#include <tf/transform_listener.h>

#include <exception>
#include <iostream>

//#include <ctime>
#ifdef GEN_PYTHON_BINDINGS
#include <costar_task_plan/utils/python.hpp>
#endif

#define _DEBUG_OUTPUT 0

using namespace dmp;

using planning_scene::PlanningScene;
using robot_model_loader::RobotModelLoader;
using robot_model::RobotModelPtr;
using robot_model::JointModel;
using robot_state::RobotState;
using collision_detection::CollisionRobot;
using planning_scene_monitor::PlanningSceneMonitor;
using planning_scene_monitor::PlanningSceneMonitorPtr;

namespace costar {

  const std::string CostarPlanner::TIME("time");
  const std::string CostarPlanner::GRIPPER("gripper");
  const std::string CostarPlanner::PS_TOPIC("monitored_planning_scene");

  const std::vector<double> &CostarPlanner::currentPos() const {
    boost::recursive_mutex::scoped_lock lock(*js_mutex);
    return x0;
  }

  const std::vector<double> &CostarPlanner::currentVel() const {
    boost::recursive_mutex::scoped_lock lock(*js_mutex);
    return x0_dot;
  }

  // Keep robot joints up to date.
  void CostarPlanner::JointStateCallback(const sensor_msgs::JointState::ConstPtr &msg) {
    boost::recursive_mutex::scoped_lock lock(*js_mutex);

    for (unsigned int i = 0; i < dof; ++i) {
      x0[i] = msg->position[i];
      if (msg->velocity.size() != 0)  {
        x0_dot[i] = msg->velocity[i];
      }
    }
  }

  // Keep the planning scene up to date. This should track the robot's current
  // position and the position of any dynamic obstacles we may need to consider.
  void CostarPlanner::PlanningSceneCallback(const moveit_msgs::PlanningScene::ConstPtr &msg) {
    boost::recursive_mutex::scoped_lock lock(*ps_mutex);
    if (msg->is_diff) {
      scene->setPlanningSceneDiffMsg(*msg);
    } else {
      scene->setPlanningSceneMsg(*msg);
    }
  }

  CostarPlanner::CostarPlanner(const std::string &robot_description_,
                           const std::string &js_topic,
                           const std::string &scene_topic,
                           const double padding,
                           const unsigned int num_basis_,
                           bool verbose_)

    : nh(), dof(7), num_basis(num_basis_), goal(7), x0(7), x0_dot(7),
    goal_threshold(7,0.1), threshold(0.1), verbose(verbose_),
    entry_names(), joint_names(7)
    {
      // needs to set up the Robot objects and listeners
      try {

        std::vector<std::string> tmp_entry_names;
        robot_model_loader = robot_model_loader::RobotModelLoaderPtr(
            new robot_model_loader::RobotModelLoader(robot_description_));
        model = robot_model::RobotModelPtr(robot_model_loader->getModel());

        ROS_INFO("Loaded model from \"%s\"!",robot_description_.c_str());

        scene = new PlanningScene(model);
        scene->getAllowedCollisionMatrix().getAllEntryNames(tmp_entry_names);
        scene->getCollisionRobotNonConst()->setPadding(padding);
        scene->propogateRobotPadding();

        for (const std::string &entry: tmp_entry_names) {
          entry_names.push_back(std::string(entry));
        }

      } catch (std::exception ex) {
        std::cerr << ex.what() << std::endl;
      }

      //js_sub = nh.subscribe(js_topic.c_str(),1000,&CostarPlanner::JointStateCallback,this);
      //ps_sub = nh.subscribe(scene_topic.c_str(),1000,&CostarPlanner::PlanningSceneCallback,this);
    }

  /* destructor */
  CostarPlanner::~CostarPlanner() {
    js_sub.shutdown();
    ps_sub.shutdown();

    delete scene;
  }

  /* add an object to the action here */
  bool CostarPlanner::AddObject(const std::string &object_name) {
    ROS_WARN("\"CostarPlanner::AddObject\" not yet implemented!");
    return false;
  }

  /* add an action */
  bool CostarPlanner::AddAction(const std::string &action_name) {
    ROS_WARN("\"CostarPlanner::AddAction\" not yet implemented!");
    return false;
  }

  /* instantiate a planning request with the given objects */
  bool CostarPlanner::Plan(const std::string &action1,
                         const std::string &action2,
                         const std::unordered_map<std::string, std::string> &object_mapping)
  {
    ROS_WARN("\"CostarPlanner::Plan\" not yet implemented!");
    return false;
  }

  void CostarPlanner::PrintInfo() const {

    //moveit_msgs::PlanningScene ps_msg;
    //monitor->getPlanningScene()->getPlanningSceneMsg(ps_msg);
    //scene->setPlanningSceneMsg(ps_msg);
    scene->getCurrentStateNonConst().update(); 

    std::vector<std::string> names = scene->getWorld()->getObjectIds();

    collision_detection::CollisionRobotConstPtr robot1 = scene->getCollisionRobot();
    std::string name = robot1->getRobotModel()->getName();

    std::cout << "==========================" << std::endl;
    std::cout << "OBJECTS IN WORLD: " << std::endl;
    for (const std::string &name: names) {
      std::cout << " -- " << name << std::endl;
    }
    std::cout << "--------------------------" << std::endl;
    std:: cout << name << std::endl;
    scene->getCurrentState().printStateInfo(std::cout);

    bool colliding = scene->isStateColliding(scene->getCurrentState(),"",true);
    std::cout << "Colliding: " << colliding << std::endl;

    std::cout << "--------------------------" << std::endl;

    std::cout << "Collisions: " << std::endl;

    scene->getAllowedCollisionMatrix().print(std::cout);
    std::cout << "==========================" << std::endl;
  }

  /* try a set of motion primitives; see if they work.
   * returns an empty trajectory if no valid path was found. */
  Traj_t CostarPlanner::TryPrimitives(std::vector<double> primitives) {
    boost::recursive_mutex::scoped_lock lock(*ps_mutex);
    scene->getCurrentStateNonConst().update(); 

    Traj_t traj;
    bool colliding, bounds_satisfied;

    collision_detection::CollisionRobotConstPtr robot1 = scene->getCollisionRobot();
    std::string name = robot1->getRobotModel()->getName();

    RobotState search_state = scene->getCurrentState();

    std::vector<DMPData> dmp_list;

    unsigned int idx = 0; // where are we reading from in the primitives
    // read out the goal
    if (verbose) {
      std::cout << "Goal: ";
    }
    for (; idx < dof; ++idx) {
      goal[idx] = primitives[idx];
      if (verbose) {
        std::cout << primitives[idx] << " ";
      }
    }
    if (verbose) {
      std::cout << std::endl;
    }
    // read out the weights
    for (unsigned int i=0; i < dof; ++i) {
      dmp::DMPData dmp_;
      dmp_.k_gain = k_gain;
      dmp_.d_gain = d_gain;

      if (verbose) {
        std::cout << "Primitive " << i << ": ";
      }
      for (unsigned int j=0; j < num_basis; ++j) {
        if (verbose) {
          std::cout << primitives[idx] << " ";
        }
        dmp_.weights.push_back(primitives[idx++]);  
      }
      if (verbose) {
        std::cout << std::endl;
      }

      dmp_list.push_back(dmp_);
    }

    unsigned char at_goal;
    DMPTraj plan;
    dmp::generatePlan(dmp_list,x0,x0_dot,0,goal,goal_threshold,-1,tau,0.1,5,plan,at_goal);

    if (verbose) {
      std::cout << "--------------------------" << std::endl;
      std::cout << "at goal: " << (unsigned int)at_goal << std::endl;
      std::cout << "points: " << plan.points.size() << std::endl;
    }

    bool drop_trajectory = false;
    for (DMPPoint &pt: plan.points) {
      Traj_pt_t traj_pt;
      traj_pt.positions = pt.positions;
      traj_pt.velocities = pt.velocities;

      if (verbose) {
        std::cout << "pt: ";
        for (double &q: pt.positions) {
          std::cout << q << " ";
        }
      }

      search_state.setVariablePositions(joint_names,traj_pt.positions);
      search_state.setVariableVelocities(joint_names,traj_pt.velocities);
      search_state.update(true);

      drop_trajectory |= !scene->isStateValid(search_state,"",verbose);

      if (verbose) {
        std::cout << " = dropped? " << drop_trajectory << std::endl;
      }


      if (drop_trajectory) {
        break;
      } else {
        traj.points.push_back(traj_pt);
      }
    }

    if (verbose) {
      std::cout << "==========================" << std::endl;
    }

    if (drop_trajectory) {
      return Traj_t(); // return an empty trajectory
    } else {
      return traj;
    }
  }

  /* update planning scene topic */
  void  CostarPlanner::SetPlanningSceneTopic(const std::string &topic) {
    //monitor->startSceneMonitor(topic);
    ROS_WARN("\"CostarPlanner::SetPlanningSceneTopic\" not currently implemented!");
  }

  /* configure degrees of freedom */
  void CostarPlanner::SetDof(const unsigned int dof_) {
    dof = dof_;
    joint_names.resize(dof);
    goal.resize(dof);
    x0.resize(dof);
    x0_dot.resize(dof);
    goal_threshold = std::vector<double>(dof,threshold);

    int i = 0;
    for (const std::string &name: scene->getCurrentState().getVariableNames()) {
      std::cout << "setting up joint " << i << ":" << name << std::endl;
      joint_names[i] = std::string(name);
      i++;
      if (i >= dof) { break; }
    }
  }

  /* configure number of basis functions */
  void CostarPlanner::SetNumBasisFunctions(const unsigned int num_) {
    num_basis = num_;
  }

  void CostarPlanner::SetK(const double k_gain_) {
    k_gain = k_gain_;
  }

  void CostarPlanner::SetD(const double d_gain_) {
    d_gain = d_gain_;
  }

  void CostarPlanner::SetTau(const double tau_) {
    tau = tau_;
  }

  void CostarPlanner::SetGoalThreshold(const double threshold_) {
    threshold = threshold_;
    goal_threshold = std::vector<double>(dof,threshold);
  }

  void CostarPlanner::SetVerbose(const bool verbose_) {
    verbose = verbose_;
  }


  /* Are we allowed to collide? */
  void CostarPlanner::SetCollisions(const std::string obj, bool allowed) {
    //std::vector<std::string> tmp;
    //scene->getAllowedCollisionMatrixNonConst().getAllEntryNames(tmp);
    //for (std::string &entry: tmp) {
    //  std::cout << entry << "\n";
    //}
    scene->getAllowedCollisionMatrixNonConst().setEntry(obj,allowed);
    //scene->getAllowedCollisionMatrixNonConst().setDefaultEntry(obj,allowed);
    //scene->getAllowedCollisionMatrixNonConst().print(std::cout);
  }

  /* Robot object default entry */
  void CostarPlanner::SetDefaultCollisions(const std::string link, bool ignore) {
    scene->getAllowedCollisionMatrixNonConst().setDefaultEntry(link, ignore);
  }

  /* try a single trajectory and see if it works. */
  bool CostarPlanner::TryTrajectory(const std::vector <std::vector<double> > &traj) {
    boost::recursive_mutex::scoped_lock lock(*ps_mutex);
    scene->getCurrentStateNonConst().update(); 

    bool colliding, bounds_satisfied;

    collision_detection::CollisionRobotConstPtr robot1 = scene->getCollisionRobot();
    std::string name = robot1->getRobotModel()->getName();

    RobotState search_state = scene->getCurrentState();

    bool drop_trajectory = false;
    for (const std::vector<double> &positions: traj) {
      if (verbose) {
        std::cout << "pt: ";
        for (double q: positions) {
          std::cout << q << " ";
        }
      }

      //search_state->setVariablePositions(joint_names,positions);
      //search_state->update(true);
      search_state.setVariablePositions(joint_names,positions);
      search_state.update(true);

      drop_trajectory |= !scene->isStateValid(search_state,"",verbose);

      if (verbose) {
        std::cout << " = dropped? " << drop_trajectory << std::endl;
      }

      if (drop_trajectory) {
        break;
      }
    }

    return !drop_trajectory;
  }

  // Try a single trajectory and see if it works.
  // This is the joint trajectory version (so we can use a consistent message type)
  bool CostarPlanner::TryTrajectory(const Traj_t &traj, unsigned int step) {
    boost::recursive_mutex::scoped_lock lock(*ps_mutex);
    scene->getCurrentStateNonConst().update(); 

    bool colliding, bounds_satisfied;

    collision_detection::CollisionRobotConstPtr robot1 = scene->getCollisionRobot();
    std::string name = robot1->getRobotModel()->getName();

    RobotState search_state = scene->getCurrentState();

    bool drop_trajectory = false;
    for (unsigned int i = 0; i < traj.points.size(); i += step) {
      const auto &pt = traj.points.at(i);
      if (verbose) {
        std::cout << "pt: ";
        for (double q: pt.positions) {
          std::cout << q << " ";
        }
      }

      // Check with the (unfinished?) collision map class. We only set this up
      // so we don't need to have a lot of repeated calls to the same or very
      // similar joint states.
      int check_result = cm.check(pt.positions);
      if (check_result > -1) {
        // If we already observed a collision during some other call this
        // iteration, then we know this is a failure already.
        drop_trajectory |= (check_result == 1);
      } else {
        // Check based on the search state.
        search_state.setVariablePositions(joint_names,pt.positions);
        search_state.update(true);

        drop_trajectory |= !scene->isStateValid(search_state,"",verbose);
      }
      if (verbose) {
        std::cout << " = dropped? " << drop_trajectory << std::endl;
      }

      // Stop checking things in this trajectory, since we already know this
      // was a failure.
      if (drop_trajectory) {
        break;
      }
    }

    // True if the trajectory never hit an invalid state.
    return !drop_trajectory;
  }


  // Reset all entries in the collision map
  void CostarPlanner::ResetCollisionMap() {
    cm.reset();
  }


#ifdef GEN_PYTHON_BINDINGS

  // Helper function: get current joint positions (according to the joint
  // listener, not to the planning scene).
  boost::python::list CostarPlanner::GetJointPositions() const {
    boost::python::list res;
    for (double x: x0) {
      res.append<double>(x);
    }
    return res;
  }

  // Try a set of motion primitives; see if they work.
  // This is aimed at the python version of the code. It takes a python list
  // containing the parameters of the motion as its only argument.
  boost::python::list CostarPlanner::pyTryPrimitives(const boost::python::list &list) {
    std::vector<double> primitives = to_std_vector<double>(list);

    scene->getCurrentStateNonConst().update(); 

#if _DEBUG_OUTPUT
    std::vector<std::string> names = scene->getWorld()->getObjectIds();

    std::cout << "==========================" << std::endl;
    std::cout << "OBJECTS IN WORLD: " << std::endl;
    for (const std::string &name: names) {
      std::cout << " -- " << name << std::endl;
    }
    std::cout << "==========================" << std::endl;
#endif

    Traj_t traj = TryPrimitives(primitives);

    boost::python::list res;

    for (const Traj_pt_t &pt: traj.points) {
      boost::python::list p;
      boost::python::list v;

      //for (const double &q: pt.positions) {
      for (unsigned int i = 0; i < pt.positions.size(); ++i) {
        p.append<double>(pt.positions[i]);
        v.append<double>(pt.velocities[i]);
      }

      res.append<boost::python::tuple>(boost::python::make_tuple(p,v));
    }

    return res;
    }

    /* try a single trajectory and see if it works.
     * this is aimed at the python version of the code. */
    bool CostarPlanner::pyTryTrajectory(const boost::python::list &trajectory) {
      std::vector< std::vector<double> > traj;
      std::vector< boost::python::list > tmp = to_std_vector<boost::python::list>(trajectory);

      for (boost::python::list &pt: tmp) {
        traj.push_back(to_std_vector<double>(pt));
      }

      return TryTrajectory(traj);
    }
#endif


  }

#ifdef GEN_PYTHON_BINDINGS
  BOOST_PYTHON_MODULE(pycostar_planner) {
    class_<costar::CostarPlanner>("CostarPlanner",init<std::string,std::string,std::string,double>())
      .def("Plan", &costar::CostarPlanner::Plan)
      .def("AddAction", &costar::CostarPlanner::AddAction)
      .def("AddObject", &costar::CostarPlanner::AddObject)
      .def("TryPrimitives", &costar::CostarPlanner::pyTryPrimitives)
      .def("TryTrajectory", &costar::CostarPlanner::pyTryTrajectory)
      .def("SetK", &costar::CostarPlanner::SetK)
      .def("SetD", &costar::CostarPlanner::SetD)
      .def("SetTau", &costar::CostarPlanner::SetTau)
      .def("SetDof", &costar::CostarPlanner::SetDof)
      .def("SetNumBasisFunctions", &costar::CostarPlanner::SetNumBasisFunctions)
      .def("SetGoalThreshold", &costar::CostarPlanner::SetGoalThreshold)
      .def("SetVerbose", &costar::CostarPlanner::SetVerbose)
      .def("PrintInfo", &costar::CostarPlanner::PrintInfo)
      .def("GetJointPositions", &costar::CostarPlanner::GetJointPositions)
      .def("SetCollisions", &costar::CostarPlanner::SetCollisions);
  }
#endif
