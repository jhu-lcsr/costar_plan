
#include <grid/dmp_trajectory_distribution.h>
#include <grid/test_features.h>
#include <grid/wam_training_features.h>
#include <grid/visualize.h>
#include <grid/grid_planner.h>
#include <grid/utils/params.h>

#include <grid/wam/input.h>

#include <trajectory_msgs/JointTrajectory.h>

#include <std_srvs/Empty.h>

using namespace grid;
using namespace KDL;

using trajectory_msgs::JointTrajectory;

int main(int argc, char **argv) {
  ros::init(argc,argv,"grid_execution_test_node");
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

  ros::ServiceClient client = nh.serviceClient<std_srvs::Empty>("/gazebo/publish_planning_scene");
  std_srvs::Empty empty;
  client.call(empty);

  TestFeatures test;
  test.addFeature("node",grid::POSE_FEATURE);
  test.addFeature("link",grid::POSE_FEATURE);
  test.addFeature("time",grid::TIME_FEATURE);
  test.setAgentFrame("wam/wrist_palm_link");
  //test.setBaseFrame("wam/base_link");
  //test.setWorldFrame("world");
  test.setWorldFrame("wam/base_link");
  test.setFrame("gbeam_node_1/gbeam_node","node");
  test.setFrame("gbeam_link_1/gbeam_link","link");

  Params p = readRosParams();

  Skill approach("approach");
  Skill grasp("grasp");
  Skill disengage("disengage");
  Skill align("align");

  approach.appendFeature("link").appendFeature("time").setInitializationFeature("link");
  grasp.appendFeature("link").appendFeature("time").setInitializationFeature("link");
  disengage.appendFeature("link").appendFeature("time").setInitializationFeature("link");
  align.appendFeature("node").appendFeature("time").setInitializationFeature("node").attachObject("link");

  if (p.skill_name == "approach") {
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
  } else if (p.skill_name == "disengage") {
    /* LOAD TRAINING DATA FOR DISENGAGE */
    std::string filenames[] = {"data/sim/disengage1.bag", "data/sim/disengage2.bag", "data/sim/disengage3.bag"};
    load_and_train_skill(disengage, rk_ptr, filenames);
  } else if (p.skill_name == "align") {
    /* LOAD TRAINING DATA FOR ALIGN*/
    std::string filenames[] = {"data/sim/align1.bag", "data/sim/align2.bag", "data/sim/align3.bag"};
    load_and_train_skill(align, rk_ptr, filenames);
    test.attachObjectFrame(align.getAttachedObject());
  }


  ROS_INFO("Done setting up. Sleeping...");
  ros::Duration(1.0).sleep();

  ros::Rate rate(1);

  ros::spinOnce();

  ROS_INFO("Updating world...");
  test.updateWorldfromTF();

  ROS_INFO("Initializing trajectory distribution...");
  DmpTrajectoryDistribution dist(rk_ptr->getDegreesOfFreedom(),5,rk_ptr);

  if (p.skill_name == "disengage") {
    dist.initialize(test,disengage);
  } else if (p.skill_name == "align") {
    dist.initialize(test,align);
  } else {
    dist.initialize(test,approach);
  }

  std::vector<EigenVectornd> params(p.ntrajs);
  std::vector<JointTrajectory> trajs(p.ntrajs);
  std::vector<double> ps(p.ntrajs);

  std::vector<trajectory_msgs::JointTrajectoryPoint> starts(p.ntrajs);

  double best_p = 0;
  unsigned int best_idx = 0;

  std::vector<double> iter_lls(p.iter);

  double model_norm = p.base_model_norm;


  if (p.detect_collisions) {
    dist.setCollisionChecker(&gp);
  }

  for (int i = 0; i < p.iter; ++i) {

    for (auto &pt: starts) {
      pt.positions = gp.currentPos();
      pt.velocities = gp.currentVel();
    }

    ros::Duration(p.wait).sleep();
    ros::spinOnce();
    rk_ptr->updateHint(gp.currentPos());
    rk_ptr->updateVelocityHint(gp.currentVel());

    // sample trajectories
    dist.sample(starts,params,trajs);
    pub.publish(toPoseArray(trajs,test.getWorldFrame(),rk_ptr));

    double sum = 0;

    // compute probabilities
    for (unsigned int j = 0; j < trajs.size(); ++j) {

      std::vector<Pose> poses = rk_ptr->FkPos(trajs[j]);

      bool collision = false;
#if 0
      if (p.detect_collisions) {
        collision = !gp.TryTrajectory(trajs[j]);
      }
#endif

      if (collision) {
        std::cout << "COLLISION DETECTED\n";
        ps[j] = 0;
      } else if (p.skill_name == "disengage") {

        disengage.resetModel();
        disengage.addModelNormalization(model_norm);

        std::vector<FeatureVector> features;
        test.getFeaturesForTrajectory(features, disengage.getFeatures(),poses);
        disengage.normalizeData(features);
        FeatureVector v = disengage.logL(features);
        ps[j] = (v.array().exp().sum() / v.size()); // would add other terms first

      } else if (p.skill_name == "align") {

        align.resetModel();
        align.addModelNormalization(model_norm);

        std::vector<FeatureVector> features;
        test.getFeaturesForTrajectory(features,align.getFeatures(),poses);
        align.normalizeData(features);
        FeatureVector v = align.logL(features);
        ps[j] = (v.array().exp().sum() / v.size()); // would add other terms first


      } else if (p.skill_name == "prepare") {

        approach.resetModel();
        approach.addModelNormalization(model_norm);

        std::vector<FeatureVector> features;
        test.getFeaturesForTrajectory(features,approach.getFeatures(),poses);

        test.setAll(features,approach.getFeatures(),"time",0);

        approach.normalizeData(features);

        FeatureVector v = approach.logL(features);
        ps[j] = (v.array().exp()(v.size()-1)); // would add other terms first

      } else {

        approach.resetModel();
        approach.addModelNormalization(model_norm);

        grasp.resetModel();
        grasp.addModelNormalization(model_norm);

        std::vector<FeatureVector> features;
        test.getFeaturesForTrajectory(features,approach.getFeatures(),poses);
        std::vector<FeatureVector> grasp_features;
        test.getFeaturesForTrajectory(grasp_features,grasp.getFeatures(),poses);

        test.setAll(grasp_features,grasp.getFeatures(),"time",0);

        approach.normalizeData(features);
        grasp.normalizeData(grasp_features);

        FeatureVector v = approach.logL(features);
        FeatureVector ve = grasp.logL(grasp_features); // gets log likelihood only for the final entry in the trajectory
        //ps[j] = (v.array().exp().sum() / v.size());// * (ve.array().exp()(ve.size()-1)); // would add other terms first
        ps[j] = (v.array().exp().sum() / v.size()) * (ve.array().exp()(ve.size()-1)); // would add other terms first
        //std::cout << ps[j] << std::endl;
      }
      sum += ps[j];

      if (ps[j] > best_p) {
        best_p = ps[j];
        best_idx = j;
      }

    }

    //if (sum > 1e-50) { 
    // update distribution
    dist.update(params,ps,p.ntrajs,p.noise,p.step_size);
    //} else {
    //i--;
    //   continue;
    //}

    std::cout << "[" << i << "] >>>> AVG P = " << (sum / p.ntrajs) << std::endl;
    iter_lls[i] = sum / p.ntrajs;


    if (i > 0 && iter_lls[i] > iter_lls[i-1]) {
      //std::cout<<"decreasing normalization\n";
      model_norm *= p.model_norm_step;
    }
  }

  std::cout << "Found best tajectory after " << p.iter << " iterations." << std::endl;

  std::cout << "Length: " << trajs[best_idx].points.size() << std::endl;
  std::cout << "DOF: " << trajs[best_idx].points[0].positions.size() << std::endl;

  // set final point to all zeros
  for (double &d: trajs[best_idx].points.rbegin()->velocities) {
    d = 0;
  }

  jpub.publish(trajs[best_idx]);
}
