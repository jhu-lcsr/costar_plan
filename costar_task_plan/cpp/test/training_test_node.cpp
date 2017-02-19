#include <ros/ros.h>
#include <costar_task_plan/training_features.h>
#include <costar_task_plan/wam_training_features.h>
#include <costar_task_plan/visualize.h>
#include <costar_task_plan/skill.h>

#include <fstream>

#include <costar_task_plan/utils/params.h>

using namespace costar;

int main(int argc, char **argv) {
  ros::init(argc,argv,"training_test_node");

  Params p = readRosParams();

  std::vector<std::string> objects;
  objects.push_back("link");
  objects.push_back("node");

  RobotKinematics *rk = new RobotKinematics("robot_description","wam/base_link","wam/wrist_palm_link");
  RobotKinematicsPtr rk_ptr = RobotKinematicsPtr(rk);

  std::vector<std::string> filenames;

  unsigned int ntraining = 3u; //9u;
  if (p.skill_name == "approach") {
    ROS_INFO("Configuring for approach...");
    std::string _filenames[] = {
      //"data/sim/old data/app1.bag", "data/sim/old data/app2.bag", "data/sim/old data/app3.bag",
      "data/sim/approach01.bag", "data/sim/approach02.bag", "data/sim/approach03.bag",
      //"data/sim_auto/approach1.bag", "data/sim_auto/approach2.bag",
      //"data/sim_auto/approach3.bag",
      //"data/sim_auto/approach4.bag",
      //"data/sim_auto/approach5.bag",
      "data/sim/approach_left01.bag",
      "data/sim/approach_left02.bag",
      "data/sim/approach_left03.bag",
      "data/sim/approach_right01.bag",
      "data/sim/approach_right02.bag",
      "data/sim/approach_right03.bag"
    };
    ntraining = 9u;
    filenames.insert(filenames.begin(),&_filenames[0],&_filenames[ntraining]);
  } else if (p.skill_name == "align") {
    ROS_INFO("Configuring for align...");
    std::string _filenames[] = {"data/sim/align1.bag", "data/sim/align2.bag", "data/sim/align3.bag",
      "data/sim_auto/align3.bag",
      "data/sim_auto/align4.bag",
      "data/sim_auto/align5.bag"
    };
    ntraining = 6u;
    filenames.insert(filenames.begin(),&_filenames[0],&_filenames[ntraining]);
  } else {
    ROS_INFO("Configuring for place...");
    std::string _filenames[] = {"data/sim/place1.bag", //"data/sim/place2.bag",
      "data/sim/place3.bag",
      "data/sim_auto/place3.bag",
      "data/sim_auto/place4.bag",
      "data/sim_auto/place5.bag"
    };
    ntraining = 5u;
    filenames.insert(filenames.begin(),&_filenames[0],&_filenames[ntraining]);
  }
  //std::string filenames[] = {"data/sim/release1b.bag", "data/sim/release2b.bag", "data/sim/release3b.bag"};
  //std::string filenames[] = {"data/sim/release1.bag", "data/sim/release2.bag", "data/sim/release3.bag"};
  //std::string filenames[] = {"data/sim/release1.bag", "data/sim/release2.bag", "data/sim/release3.bag",
  //  "data/sim/release1b.bag", "data/sim/release2b.bag", "data/sim/release3b.bag","data/sim/release1c.bag", "data/sim/release2c.bag", "data/sim/release3c.bag"};
  //std::string filenames[] = {"data/sim/grasp1.bag", "data/sim/grasp2.bag", "data/sim/grasp3.bag"};

  std::vector<std::shared_ptr<WamTrainingFeatures> > wtf(ntraining);

  for (unsigned int i = 0; i < ntraining; ++i) {
    std::shared_ptr<WamTrainingFeatures> wtf_ex(new WamTrainingFeatures(objects));
    wtf_ex->addFeature("time",TIME_FEATURE);
    wtf_ex->setRobotKinematics(rk_ptr);
    wtf_ex->read(filenames[i],10);
    if (p.skill_name != "approach") {
      ROS_INFO("attaching link object");
      wtf_ex->attachObjectFrame("link");
    }
    wtf[i] = wtf_ex;
  }

  wtf[0]->printTrainingFeaturesInfo();
  wtf[0]->printExtractedFeatures();

  std::vector<FeatureVector> data;
  std::cout << "Getting features..." << std::endl;
  {

    std::vector<std::string> features;
    if (p.skill_name == "approach") {
      ROS_INFO("Using link features");
      features.push_back("link");
    } else {
      ROS_INFO("Using node features");
      features.push_back("node");
    }
    features.push_back("time");

    clock_t begin = clock();
    for (unsigned int i = 0; i < ntraining; ++i) {
      std::vector<FeatureVector> ex_data = wtf[i]->getFeatureValues(features);
      //std::cout << "... preparing example " << (i+1) << " with " << ex_data.size() << " observations." << std::endl;
      data.insert(data.end(),ex_data.begin(),ex_data.end());
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Converting into features took " << elapsed_secs << "seconds." << std::endl;
  }

  std::cout << "Total observations: " << data.size() << std::endl;
  std::vector<std::pair<FeatureVector,double> > training_data;
  unsigned int size = data[0].size();
  for (FeatureVector &vec: data) {
    std::pair<FeatureVector,double> obs(vec,1.0/data.size());
    //for (unsigned int i = 0; i < vec.size(); ++i) {
    //  std::cout << vec(i) << " ";
    //}
    //std::cout << std::endl;
    if (size != vec.size()) {
      std::cout << "ERROR: " << size << " != " << vec.size() << "!" << std::endl;
      break;
    }
    training_data.push_back(obs);
  }

  // try learning a GMM model
  std::cout << "... converted into training data with " << data.size() << " weighted observations." << std::endl;
  Gmm gmm(size,1);
  gmm.Init(*data.begin(),*data.rbegin());
  //gmm.Print(std::cout);

  {
    clock_t begin = clock();
    gmm.Fit(training_data);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Fitting GMM took " << elapsed_secs << "seconds." << std::endl;
  }


  std::cout << "Successfully fit GMM!" << std::endl;
  gmm.Print(std::cout);

  std::cout << "Running skill test:" << std::endl;

  geometry_msgs::PoseArray msg;
  Skill test(p.skill_name,3);
  if (p.skill_name == "approach") {
    ROS_INFO("Using link features");
    msg.header.frame_id = "gbeam_link_1/gbeam_link"; //"wam/wrist_palm_link";
    test.appendFeature("link").appendFeature("time");
  } else {
    ROS_INFO("Using node features");
    test.appendFeature("node").appendFeature("time");
    msg.header.frame_id = "gbeam_node_1/gbeam_node";//"gbeam_link_1/gbeam_link"; //"wam/wrist_palm_link";
  }

  for (unsigned int i = 0; i < ntraining; ++i) {
    std::cout << " ... " << i << "\n";
    test.addTrainingData(*wtf[i]);
  }
  test.trainSkillModel();
  std::cout << "Skill trained!" << std::endl;
  test.printGmm();

  // publish trajectories
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples",1000);

  // get ready
  //msg.header.frame_id = "gbeam_node_1/gbeam_node";//"gbeam_link_1/gbeam_link"; //"wam/wrist_palm_link";
  //msg.header.frame_id = "wam/wrist_palm_link";
  for (unsigned int i = 0; i < ntraining; ++i) {

    std::vector<Pose> poses;
    if (p.skill_name == "approach") {
      poses = wtf[i]->getPose("link");
    } else {
      poses = wtf[i]->getPose("node");
    }
    std::vector<FeatureVector> v = wtf[i]->getFeatureValues(test.getFeatures());

    //for (Pose &pose: poses) {
    for (FeatureVector &vec: v) {
      Pose pose = wtf[i]->getPoseFrom("link",vec);
      for (unsigned int i = 0; i < vec.size(); ++ i) {
        std::cout << vec(i) << " ";
      }
      std::cout << "\n";
      tf::Pose tfp;
      geometry_msgs::Pose p;
      tf::poseKDLToTF(pose,tfp);
      tf::poseTFToMsg(tfp,p);
      msg.poses.push_back(p);
    }



    test.normalizeData(v);
    FeatureVector p = test.logL(v);
    std::cout << "[" << i << "] avg = " << p.sum() / p.size() << std::endl;

  }

  ros::Rate rate = ros::Rate(10);
  while (ros::ok()) {

    pub.publish(msg);
    ros::spinOnce();
    rate.sleep();
  }

}
