
#include <costar_task_plan/trajectory_distribution.h>
#include <costar_task_plan/test_features.h>
#include <costar_task_plan/wam_training_features.h>
#include <costar_task_plan/visualize.h>

using namespace costar;
using namespace KDL;

int main(int argc, char **argv) {
  ros::init(argc,argv,"costar_trajectory_test_node");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples",1000);

  RobotKinematicsPtr rk_ptr = RobotKinematicsPtr(new RobotKinematics("robot_description","wam/base_link","wam/wrist_palm_link"));

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

  Skill approach("approach",1);
  approach.appendFeature("link").appendFeature("time");
  //approach.appendFeature("time");
  
  // set feature to use to initialize this action
  // must be a pose so we can find out where to start looking
  approach.setInitializationFeature("link"); 

  /* LOAD TRAINING DATA */
  {

    std::vector<std::string> objects;
    objects.push_back("link");
    objects.push_back("node");

    std::string filenames[] = {"data/sim/app1.bag", "data/sim/app2.bag", "data/sim/app3.bag"};

    std::vector<std::shared_ptr<WamTrainingFeatures> > wtf(3);
    for (unsigned int i = 0; i < 3; ++i) {
      std::shared_ptr<WamTrainingFeatures> wtf_ex(new WamTrainingFeatures(objects));
      wtf_ex->addFeature("time",TIME_FEATURE);
      wtf_ex->setRobotKinematics(rk_ptr);
      wtf_ex->read(filenames[i]);
      wtf[i] = wtf_ex;
    }

    // add each skill
    for (unsigned int i = 0; i < 3; ++i) {
      approach.addTrainingData(*wtf[i]);
    }
    approach.trainSkillModel();
    approach.printGmm();
  
    for (unsigned int i = 0; i < 3; ++i) {
      std::shared_ptr<WamTrainingFeatures> wtf_ex(new WamTrainingFeatures(objects));
      wtf_ex->addFeature("time",TIME_FEATURE); // add time as a feature
      wtf_ex->setRobotKinematics(rk_ptr); // set kinematics provider
      wtf_ex->read(filenames[i]); // load data
      std::vector<FeatureVector> data = wtf_ex->getFeatureValues(approach.getFeatures()); // get data
      approach.normalizeData(data);
      FeatureVector v = approach.logL(data); // get log likelihoods in an Eigen vector
      double p = v.sum() / v.size();
      std::cout << "training example " << i << ": p = " << p << std::endl;
    }
  }


  /* LOOP */

  ROS_INFO("Done setting up. Sleeping...");
  ros::Duration(1.0).sleep();

  ros::Rate rate(1);
  unsigned int ntrajs = 5;
  try {
    while (ros::ok()) {

      ROS_INFO("Updating world...");
      test.updateWorldfromTF();

      ROS_INFO("Initializing trajectory distribution...");
      TrajectoryDistribution dist(3,1);
      dist.initialize(test,approach);

      ROS_INFO("Generating trajectories...");
      std::vector<Trajectory *> trajs(ntrajs);
      std::vector<EigenVectornd> params(ntrajs);

      // look at the time it takes to compute features
      {
        using namespace std;

        clock_t begin = clock();
        dist.sample(params,trajs);
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Sampling " << ntrajs << " trajectories took " << elapsed_secs << " seconds." << std::endl;
      }

      // generate the features
      // see how long that takes
      {
        using namespace std;

        std::vector<FeatureVector> features;
        clock_t begin = clock();
        for (unsigned int i = 0; i < trajs.size(); ++i) {

          test.getFeaturesForTrajectory(features,approach.getFeatures(),trajs[i]);
          approach.normalizeData(features);
          FeatureVector v = approach.logL(features);

          double p = v.sum() / v.size();
          std::cout << " - traj " << i << ": avg p = " << p << std::endl;
        }
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Generating features for " << ntrajs << " trajectories took " << elapsed_secs << " seconds." << std::endl;
      }

      // print out all the sampled trajectories
      std::cout << "Publishing trajectories... ";
      pub.publish(toPoseArray(trajs,0.05,test.getWorldFrame()));
      std::cout << "done." << std::endl;

      for(unsigned int i = 0; i < trajs.size(); ++i) {
        delete trajs[i];
      }

      rate.sleep();
    }
  } catch (ros::Exception ex) {
    ROS_ERROR("%s",ex.what());
  }

}
