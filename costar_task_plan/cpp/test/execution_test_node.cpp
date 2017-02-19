
#include <costar_task_plan/trajectory_distribution.h>
#include <costar_task_plan/test_features.h>
#include <costar_task_plan/wam_training_features.h>
#include <costar_task_plan/visualize.h>
#include <costar_task_plan/costar_planner.h>

#include <costar_task_plan/wam/input.h>

#include <trajectory_msgs/JointTrajectory.h>

using namespace costar;
using namespace KDL;

using trajectory_msgs::JointTrajectory;

int main(int argc, char **argv) {
  ros::init(argc,argv,"costar_execution_test_node");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples",1000);
  ros::Publisher jpub = nh.advertise<trajectory_msgs::JointTrajectory>("trajectory",1000);

  /* SET UP THE ROBOT KINEMATICS */
  RobotKinematicsPtr rk_ptr = RobotKinematicsPtr(new RobotKinematics("robot_description","wam/base_link","wam/wrist_palm_link"));

  GridPlanner gp("robot_description","/gazebo/barrett_manager/wam/joint_states","/gazebo/planning_scene");
  gp.SetDof(7);
  gp.SetNumBasisFunctions(5);
  gp.SetK(100);
  gp.SetD(20);
  gp.SetTau(1.0);
  gp.SetGoalThreshold(0.1);

  TestFeatures test;
  test.addFeature("node",costar::POSE_FEATURE);
  test.addFeature("link",costar::POSE_FEATURE);
  test.addFeature("time",costar::TIME_FEATURE);
  test.setAgentFrame("wam/wrist_palm_link");
  //test.setBaseFrame("wam/base_link");
  //test.setWorldFrame("world");
  test.setWorldFrame("wam/base_link");
  test.setFrame("gbeam_node_1/gbeam_node","node");
  test.setFrame("gbeam_link_1/gbeam_link","link");

  double step_size;
  double noise;
  int ntrajs = 50;
  int iter = 10;
  std::string skill_name;
  ros::NodeHandle nh_tilde("~");
  if (not nh_tilde.getParam("step_size",step_size)) {
    step_size = 0.80;
  }
  if (not nh_tilde.getParam("noise",noise)) {
    noise = 1e-10;
  }
  if (not nh_tilde.getParam("ntrajs",ntrajs)) {
    ntrajs = 50;
  }
  if (not nh_tilde.getParam("iter",iter)) {
    iter = 10;
  }
  if (not nh_tilde.getParam("skill",skill_name)) {
    skill_name = "approach";
  }

  Skill approach("approach");
  Skill grasp("grasp");
  Skill disengage("disengage");

  approach.appendFeature("link").appendFeature("time").setInitializationFeature("link");
  grasp.appendFeature("link").appendFeature("time").setInitializationFeature("link");
  disengage.appendFeature("link").appendFeature("time").setInitializationFeature("link");

  /* LOAD TRAINING DATA FOR APPROACH */
  {
    std::string filenames[] = {"data/sim/app1.bag", "data/sim/app2.bag", "data/sim/app3.bag"};
    load_and_train_skill(approach, rk_ptr, filenames);
  }
  /* LOAD TRAINING DATA FOR GRASP */
  {
    std::string filenames[] = {"data/sim/grasp1.bag", "data/sim/grasp2.bag", "data/sim/grasp3.bag"};
    load_and_train_skill(grasp, rk_ptr, filenames);
  }
  /* LOAD TRAINING DATA FOR DISENGAGE */
  {
    std::string filenames[] = {"data/sim/disengage1.bag", "data/sim/disengage2.bag", "data/sim/disengage3.bag"};
    load_and_train_skill(disengage, rk_ptr, filenames);
  }

  ROS_INFO("Done setting up. Sleeping...");
  ros::Duration(1.0).sleep();

  ros::Rate rate(1);

  ros::spinOnce();

  ROS_INFO("Updating world...");
  test.updateWorldfromTF();

  ROS_INFO("Initializing trajectory distribution...");
  TrajectoryDistribution dist(3,1);

  if (skill_name == "disengage") {
    dist.initialize(test,disengage);
  } else {
    dist.initialize(test,approach);
  }

  std::vector<Trajectory *> trajs(ntrajs);
  std::vector<EigenVectornd> params(ntrajs);
  std::vector<JointTrajectory> joint_trajs(ntrajs);
  std::vector<double> ps(ntrajs);

  for(unsigned int i = 0; i < ntrajs; ++i) {
    trajs[i] = 0;
    //joint_trajs[i] = rk_ptr->getEmptyJointTrajectory();
  }


  double best_p = 0;
  unsigned int best_idx = 0;

  for (int i = 0; i < iter; ++i) {

    for (unsigned int j = 0; j < trajs.size(); ++j) {
      if (trajs[j]) {
        delete trajs[j];
        trajs[j] = 0;
      }
    }

    //ros::Duration(0.25).sleep();
    ros::spinOnce();
    rk_ptr->updateHint(gp.currentPos());

    // sample trajectories
    dist.sample(params,trajs);
    pub.publish(toPoseArray(trajs,0.05,test.getWorldFrame()));

    double sum = 0;

    // compute probabilities
    for (unsigned int j = 0; j < trajs.size(); ++j) {
      bool res = rk_ptr->toJointTrajectory(trajs[j],joint_trajs[j],0.1);
      if (skill_name == "disengage") {

        std::vector<FeatureVector> features;
        test.getFeaturesForTrajectory(features,disengage.getFeatures(),trajs[j]);
        disengage.normalizeData(features);

        FeatureVector v = disengage.logL(features);
        ps[j] = (double)res * (v.array().exp().sum() / v.size()); // would add other terms first
      } else {

        std::vector<FeatureVector> features;
        test.getFeaturesForTrajectory(features,approach.getFeatures(),trajs[j]);
        std::vector<FeatureVector> grasp_features;
        test.getFeaturesForTrajectory(grasp_features,grasp.getFeatures(),trajs[j]);

        test.setAll(grasp_features,grasp.getFeatures(),"time",0);

        approach.normalizeData(features);
        grasp.normalizeData(grasp_features);

        FeatureVector v = approach.logL(features);
        FeatureVector ve = grasp.logL(grasp_features); // gets log likelihood only for the final entry in the trajectory
        ps[j] = (double)res * (v.array().exp().sum() / v.size()) * (ve.array().exp()(ve.size()-1)); // would add other terms first
        //ps[j] = (double)res * (v.array().sum() / v.size()) + (ve.array()(ve.size()-1)); // would add other terms first
      }
      sum += ps[j];

      if (ps[j] > best_p) {
        best_p = ps[j];
        best_idx = j;
      }
    }

    if (sum > 1e-50) { 
      // update distribution
      dist.update(params,ps,noise,step_size);
    } else {
      //i--;
      continue;
    }

    std::cout << "[" << i << "] >>>> AVG P = " << (sum / ntrajs) << std::endl;
  }

  std::cout << "Found best tajectory after " << iter << " iterations." << std::endl;

  bool res = rk_ptr->toJointTrajectory(trajs[best_idx],joint_trajs[best_idx],0.1);
  std::cout << "IK result: "; //<< res << std::endl;
  if (res) {
    std:: cout << "SUCCESS!" << std::endl;
  } else {
    std::cout << "failure :(" << std::endl;
  }

  std::cout << "Length: " << joint_trajs[best_idx].points.size() << std::endl;
  std::cout << "DOF: " << joint_trajs[best_idx].points[0].positions.size() << std::endl;

  // set final point to all zeros
  for (double &d: joint_trajs[best_idx].points.rbegin()->velocities) {
    d = 0;
  }

  if (res) {
    jpub.publish(joint_trajs[best_idx]);
  }

  for (unsigned int j = 0; j < trajs.size(); ++j) {
    delete trajs[j];
    trajs[j] = 0;
  }
}
