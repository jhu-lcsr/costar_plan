
// MoveIt!
#include <moveit/collision_detection/collision_robot.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

using planning_scene::PlanningScene;
using robot_model_loader::RobotModelLoader;
using robot_model::RobotModelPtr;
using robot_model::JointModel;
using robot_state::RobotState;
using collision_detection::CollisionRobot;
using planning_scene_monitor::PlanningSceneMonitor;
using planning_scene_monitor::PlanningSceneMonitorPtr;

int main(int argc, char **argv) {
  ros::init(argc,argv,"moveit_test");

  const std::string robot_description_ = std::string("robot_desciption");
  const std::string joint_state_topic_ = std::string("joint_states");
  const std::string planning_scene_topic_ = std::string("planning_scene");

  std::vector<std::string> tmp_entry_names;

  robot_model_loader::RobotModelLoader robot_model_loader(robot_description_);
  ROS_INFO("Loaded model from \"%s\"!",robot_description_.c_str());
  double padding = 0.;

  robot_model::RobotModelPtr model = (robot_model_loader.getModel());

  boost::shared_ptr<planning_scene::PlanningScene> scene = boost::shared_ptr<PlanningScene>(new PlanningScene(robot_model_loader.getModel()));
  scene->getAllowedCollisionMatrix().getAllEntryNames(tmp_entry_names);
  scene->getCollisionRobotNonConst()->setPadding(padding);
  scene->propogateRobotPadding();
  boost::shared_ptr<robot_state::RobotState> state = boost::shared_ptr<RobotState>(new RobotState(robot_model_loader.getModel()));
  boost::shared_ptr<robot_state::RobotState> search_state = boost::shared_ptr<RobotState>(new RobotState(robot_model_loader.getModel()));

  return 0;
}
