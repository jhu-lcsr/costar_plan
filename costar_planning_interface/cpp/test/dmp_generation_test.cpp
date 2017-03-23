
#include <costar_task_plan/test_features.h>
#include <costar_task_plan/wam_training_features.h>
#include <costar_task_plan/visualize.h>
#include <costar_task_plan/dmp_trajectory_distribution.h>

#include <costar_task_plan/costar_planner.h>

using namespace costar;
using namespace KDL;

int main(int argc, char **argv) {
  ros::init(argc,argv,"costar_trajectory_test_node");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples",1000);

  RobotKinematicsPtr rk_ptr = RobotKinematicsPtr(new RobotKinematics("robot_description","wam/base_link","wam/wrist_palm_link"));
  CostarPlanner gp("robot_description","/gazebo/barrett_manager/wam/joint_states","/gazebo/planning_scene");
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
  test.setWorldFrame("wam/base_link");
  test.setFrame("gbeam_node_1/gbeam_node","node");
  test.setFrame("gbeam_link_1/gbeam_link","link");

  Skill approach("approach",1);
  approach.appendFeature("link").appendFeature("time");

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

  ros::spinOnce();
  ROS_INFO("Done setting up. Sleeping...");
  ros::Duration(1.0).sleep();

  ros::Rate rate(1);
  unsigned int ntrajs = 200;

  std::vector<JointTrajectory> trajs;
  trajs.resize(ntrajs);

  std::vector<trajectory_msgs::JointTrajectoryPoint> starts(ntrajs);

  try {
    while (ros::ok()) {
      ros::spinOnce();

      ROS_INFO("Updating world...");
      test.updateWorldfromTF();

      rk_ptr->updateHint(gp.currentPos());
      rk_ptr->updateVelocityHint(gp.currentVel());

      ROS_INFO("Initializing trajectory distribution...");
      std::cout << rk_ptr->getDegreesOfFreedom() << std::endl;
      DmpTrajectoryDistribution dist(rk_ptr->getDegreesOfFreedom(),5,rk_ptr);
      dist.initialize(test,approach);


      for (auto &pt: starts) {
        pt.positions = gp.currentPos();
        pt.velocities = gp.currentVel();
      }

      ROS_INFO("Generating trajectories...");
      std::vector<EigenVectornd> params(ntrajs);

      // look at the time it takes to compute features
      {
        using namespace std;

        clock_t begin = clock();
        dist.sample(starts,params,trajs);
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Sampling " << ntrajs << " trajectories took " << elapsed_secs << " seconds." << std::endl;
      }

      std::vector<FeatureVector> features;
      // generate the features
      // see how long that takes
      {
        using namespace std;

        clock_t begin = clock();
        for (unsigned int i = 0; i < trajs.size(); ++i) {

          std::vector<Pose> poses = rk_ptr->FkPos(trajs[i]);
          {
            //clock_t begin = clock();
            //poses = rk_ptr->FkPos(trajs[i]);
            //clock_t end = clock();
            //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            //std::cout << " - fwd kinematics took " << elapsed_secs << " seconds." << std::endl;
          }

          test.getFeaturesForTrajectory(features,approach.getFeatures(),poses);
          approach.normalizeData(features);
          FeatureVector v = approach.logL(features);

          double p = v.sum() / v.size();
          //std::cout << " - traj " << i << ": avg p = " << p << std::endl;
        }
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Generating features for " << ntrajs << " trajectories took " << elapsed_secs << " seconds." << std::endl;
      }

      // print out all the sampled trajectories
      std::cout << "Publishing trajectories... ";
      pub.publish(toPoseArray(trajs,test.getWorldFrame(),rk_ptr));
      std::cout << "done." << std::endl;

      rate.sleep();
    }
  } catch (ros::Exception ex) {
    ROS_ERROR("%s",ex.what());
  }

}
