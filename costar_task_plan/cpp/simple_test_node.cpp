

/**
 * TASK MODEL TEST
 * This test creates the default WAM task model, plus TestFeatures object to retrieve data.
 * It creates a bunch of different versions of each of the skills for various different objects in the world.
 */


#include <costar_task_plan/skill.h>
#include <costar_task_plan/task_model.h>
#include <costar_task_plan/robot_kinematics.h>
#include <costar_task_plan/costar_planner.h>
#include <costar_task_plan/visualize.h>
#include <costar_task_plan/utils/params.h>
#include <costar_task_plan/wam/input.h>

#include "wam/load_wam_skills.hpp"
#include "auto/load_wam_skills_auto_data.hpp"

#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>

#include <costar_task_plan/CommandAction.h>

using namespace costar;

void update_features(std::unordered_map<std::string, TestFeaturesPtr> &features) {
  for (auto &nf: features) {
    nf.second->updateWorldfromTF();
  }
}

void load_to_one_array(std::vector<InstantiatedSkillPtr> &is, std::vector<JointTrajectory> &trajs) {
  trajs.resize(0);
  for (auto &ptr: is) {
    if (ptr->last_samples > 0) { // and ptr->last_probability > 1e-199) {
      for (auto &traj: ptr->trajs) {
        trajs.push_back(traj);
      }
    }
  }
}

int main(int argc, char **argv) {

  ros::init(argc,argv,"task_model_test_node");
  ros::NodeHandle nh;

  actionlib::SimpleActionClient<costar_plan_msgs::CommandAction> ac("command", true);

  Params p = readRosParams();
  RobotKinematicsPtr robot = RobotKinematicsPtr(new RobotKinematics("robot_description","wam/base_link","wam/wrist_palm_link"));
  GridPlanner gp("robot_description","/gazebo/barrett_manager/wam/joint_states","/gazebo/raw_planning_scene",0.02);
  gp.SetDof(robot->getDegreesOfFreedom());
  gp.SetCollisions("gbeam_soup",true);
  gp.SetCollisions("gbeam_soup.gbeam_link_1",true);
  gp.SetCollisions("gbeam_soup.gbeam_link_2",true);
  gp.SetCollisions("gbeam_soup.gbeam_node_1",false);
  gp.SetCollisions("gbeam_soup.gbeam_node_2",false);

  // disable a bunch of collisions
  gp.SetDefaultCollisions("wam/hand/finger_1/dist_link",true);
  gp.SetDefaultCollisions("wam/hand/finger_2/dist_link",true);
  gp.SetDefaultCollisions("wam/hand/finger_3/dist_link",true);
  gp.SetDefaultCollisions("wam/hand/finger_1/prox_link",true);
  gp.SetDefaultCollisions("wam/hand/finger_2/prox_link",true);
  gp.SetDefaultCollisions("wam/hand/finger_3/prox_link",true);
  //gp.SetDefaultCollisions("wam/shoulder_yaw_link",true);
  //gp.SetDefaultCollisions("wam/upper_arm_link",true);
  //gp.SetDefaultCollisions("wam/base_link",true);
  //gp.SetDefaultCollisions("wam/shoulder_pitch_link",true);
  //gp.SetDefaultCollisions("wam/forearm_link",true);
  gp.SetDefaultCollisions("wam/wrist_pitch_link",true);
  gp.SetDefaultCollisions("wam/wrist_yaw_link",true);
  gp.SetDefaultCollisions("wam/hand/bhand_grasp_link",true);
  gp.SetDefaultCollisions("wam/wrist_palm_link",true);

  gp.SetVerbose(p.collisions_verbose);

  GridPlanner gp2("robot_description","/gazebo/barrett_manager/wam/joint_states","/gazebo/raw_planning_scene",0.01);
  gp2.SetDof(robot->getDegreesOfFreedom());
  gp2.SetCollisions("gbeam_soup",true);

  // disable a bunch of collisions
  //gp2.SetDefaultCollisions("wam/hand/finger_1/prox_link",true);
  //gp2.SetDefaultCollisions("wam/hand/finger_2/prox_link",true);
  //gp2.SetDefaultCollisions("wam/hand/finger_3/prox_link",true);
  gp2.SetDefaultCollisions("wam/shoulder_yaw_link",true);
  //gp2.SetDefaultCollisions("wam/upper_arm_link",true);
  gp2.SetDefaultCollisions("wam/base_link",true);
  gp2.SetDefaultCollisions("wam/shoulder_pitch_link",true);
  //gp2.SetDefaultCollisions("wam/forearm_link",true);
  //gp2.SetDefaultCollisions("wam/wrist_pitch_link",true);
  //gp2.SetDefaultCollisions("wam/wrist_yaw_link",true);

  if (p.collisions_verbose) {
    gp.PrintInfo();
  }

  GridPlanner *checker = 0;
  GridPlanner *checker2 = 0;
  if (p.detect_collisions) {
    checker = &gp;
    checker2 = &gp2;
  }

  ros::ServiceClient client = nh.serviceClient<std_srvs::Empty>("/gazebo/publish_planning_scene");
  std_srvs::Empty empty;
  client.call(empty);

  std::unordered_map<std::string, SkillPtr> skills;
  if (p.test == 0) {
    skills = loadWamSkills();
  } else if (p.test == 1) {
    skills = loadWamSkillsAuto();
  }
  std::unordered_map<std::string, TestFeaturesPtr> features = setupTestFeaturesForTrials();

  InstantiatedSkillPtr root = InstantiatedSkill::Root();

  ros::Publisher pub = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples",1000);
  ros::Publisher pub2 = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples_2",1000);
  ros::Publisher pub3 = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples_3",1000);
  ros::Publisher pub4 = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples_4",1000);
  ros::Publisher pub5 = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples_5",1000);
  ros::Publisher pub6 = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples_6",1000);
  ros::Publisher attached_pub = nh.advertise<geometry_msgs::PoseArray>("trajectory_examples_attached",1000);

  ros::spinOnce();
  robot->updateHint(gp.currentPos());
  robot->updateVelocityHint(gp.currentVel());

  std::cout << "sleeping...\n";

  ros::Duration(1.0).sleep();

  ros::spinOnce();
  robot->updateHint(gp.currentPos());
  robot->updateVelocityHint(gp.currentVel());

  update_features(features);

  /*************************************************************************/

  unsigned int nbasis = 5;
  InstantiatedSkillPtr app1 = InstantiatedSkill::DmpInstance(skills.at("approach"), features.at("node1,link1"), robot, nbasis, checker2);
  InstantiatedSkillPtr app2 = InstantiatedSkill::DmpInstance(skills["approach_right"], features["node2,link2"], robot, nbasis, checker2);
  InstantiatedSkillPtr app3 = InstantiatedSkill::DmpInstance(skills["approach_left"], features["node1,link3"], robot, nbasis, checker2);

  std::cout << "Initializing grasps..." << std::endl;
  InstantiatedSkillPtr grasp1 = InstantiatedSkill::DmpInstance(skills["grasp"], features["node1,link1"], robot, nbasis, checker);
  std::cout << "Initializing aligns..." << std::endl;
  InstantiatedSkillPtr align11 = InstantiatedSkill::DmpInstance(skills["align"], features["node1,link1"], robot, nbasis, checker);
  std::cout << "Initializing places..." << std::endl;
  InstantiatedSkillPtr place11 = InstantiatedSkill::DmpInstance(skills["place"], features["node1,link1"], robot, nbasis, checker);
  std::cout << "Initializing releases..." << std::endl;
  InstantiatedSkillPtr release11 = InstantiatedSkill::DmpInstance(skills["release"], features["node1,link1"], robot, nbasis, checker);
  std::cout << "Initializing disengages..." << std::endl;
  InstantiatedSkillPtr disengage11 = InstantiatedSkill::DmpInstance(skills["disengage"], features["node1,link1"], robot, nbasis, checker);
  root->addNext(app1); 
  app1->addNext(grasp1); app1->pub = &pub;
  grasp1->addNext(align11); 
  align11->addNext(place11); align11->pub = &pub3;
  place11->addNext(release11); place11->pub = &pub4;

  std::vector<InstantiatedSkillPtr> approaches; approaches.push_back(app1);
  std::vector<InstantiatedSkillPtr> aligns; aligns.push_back(align11);
  std::vector<InstantiatedSkillPtr> releases; releases.push_back(release11);
  std::vector<InstantiatedSkillPtr> places; places.push_back(place11);
  std::vector<InstantiatedSkillPtr> grasps; grasps.push_back(grasp1);
  std::vector<InstantiatedSkillPtr> disengages; disengages.push_back(disengage11);

  /*************************************************************************/

  std::vector<trajectory_msgs::JointTrajectory> approach_trajs;
  std::vector<trajectory_msgs::JointTrajectory> disengage_trajs;
  std::vector<trajectory_msgs::JointTrajectory> align_trajs;
  std::vector<trajectory_msgs::JointTrajectory> place_trajs;
  std::vector<trajectory_msgs::JointTrajectory> grasp_trajs;
  std::vector<trajectory_msgs::JointTrajectory> release_trajs;

  std::vector<double> ps(1.0,1);
  std::vector<double> ps_out(1.0,1);
  std::vector<trajectory_msgs::JointTrajectoryPoint> starts(1);

  for (auto &pt: starts) {
    pt.positions = gp.currentPos();
    pt.velocities = gp.currentVel();
  }
  ps[0] = 1.;

  std::vector<double> iter_p(p.iter);

  int horizon = p.starting_horizon;
  double prob = 0;
  for (unsigned int i = 0; i < p.iter; ++i) { 
    assert(ros::ok());
    ros::spinOnce();
    robot->updateHint(gp.currentPos());
    robot->updateVelocityHint(gp.currentVel());


    // this is where the magic happens
    //ps_out[0] = 0.;
    //ps[0] = 1.; // set prior
    ps_out[0] = 0.;
    ps[0] = 0.; // set prior
    root->step(ps,starts,ps_out,prob,1,horizon,p.ntrajs);

    /* PUT EVERYTHING INTO SOME MESSAGES */
    {
      load_to_one_array(approaches,approach_trajs);
      load_to_one_array(aligns,align_trajs);
      load_to_one_array(places,place_trajs);
      load_to_one_array(grasps,grasp_trajs);
      load_to_one_array(disengages,disengage_trajs);
      load_to_one_array(releases,release_trajs);
      std::cout << "pub len = " << approach_trajs.size() << "\n";
      pub.publish(toPoseArray(approach_trajs,app1->features->getWorldFrame(),robot));
      pub2.publish(toPoseArray(disengage_trajs,app1->features->getWorldFrame(),robot));
      pub5.publish(toPoseArray(grasp_trajs,grasp1->features->getWorldFrame(),robot));
      pub6.publish(toPoseArray(release_trajs,app1->features->getWorldFrame(),robot));
      pub3.publish(toPoseArray(align_trajs,app1->features->getWorldFrame(),robot));
      attached_pub.publish(toPoseArray(place_trajs,app1->features->getWorldFrame(),robot,align11->features->getAttachedObjectFrame()));
      //attached_pub.publish(toPoseArray(place_trajs,app1->features->getWorldFrame(),robot,place22->getAttachedObjectFrame()));
      pub4.publish(toPoseArray(place_trajs,app1->features->getWorldFrame(),robot));
    }

    iter_p[i] = exp(ps_out[0]);
    std::cout << "\n\n===================================\n";
    std::cout << "ITER " << i; // << std::endl;
    std::cout << ": " << iter_p[i] << " ... ";
    if (i > 1) {

      std::cout << fabs(iter_p[i] - iter_p[i-1]) << " < " << (p.update_horizon * iter_p[i]);
      std::cout << std::endl;

      if (fabs(iter_p[i] - iter_p[i-1]) < (p.update_horizon * iter_p[i])) {
        ++horizon;
        std::cout << "horizon = " << horizon << "\n";
        if (horizon > p.max_horizon) {
          std::cout << std::endl;
          --horizon; // don't execute that last node
          break;
        } else {
          root->refresh(horizon-1);
        }
        //root->refresh(horizon);
      }
    } else {
      std::cout << std::endl;
    }

    ros::Duration(p.wait).sleep();
  }

  // execute here
  root->execute(gp,ac,p.execute_depth,false,p.replan_depth);
  //align22->execute(gp,ac,horizon-1,true);

}
