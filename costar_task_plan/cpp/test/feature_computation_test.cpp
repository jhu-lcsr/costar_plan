
#include <grid/test_features.h>
#include <grid/visualize.h>

#include <ros/ros.h>
#include <ctime>
#include <memory>

#include <cmath>

/* KDL includes */
#include <kdl/trajectory_composite.hpp>
#include <kdl/trajectory_segment.hpp>
#include <kdl/velocityprofile_spline.hpp>
#include <kdl/velocityprofile_trap.hpp>
#include <kdl/rotational_interpolation_sa.hpp>
#include <kdl/path_roundedcomposite.hpp>

using namespace grid;
using namespace KDL;

int main (int argc, char **argv) {
  ros::init(argc,argv,"grid_features_test_node");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<geometry_msgs::PoseArray>("trajectory",1000);
  ros::Publisher fpub = nh.advertise<geometry_msgs::PoseArray>("features",1000);

  TestFeatures test;
  test.addFeature("node",grid::POSE_FEATURE);
  test.addFeature("link",grid::POSE_FEATURE);
  test.setAgentFrame("wam/wrist_palm_link");
  test.setWorldFrame("world");
  test.setFrame("gbeam_node_1/gbeam_node","node");
  test.setFrame("gbeam_link_1/gbeam_link","link");
  test.attachObjectFrame("link");

  ROS_INFO("Done setting up. Sleeping...");
  ros::Duration(1.0).sleep();
  ROS_INFO("Transforms:");
  test.lookup("node");
  test.lookup("link");
  test.lookup("agent");

  ros::Rate rate(30);
  try {
    while (ros::ok()) {

      test.updateWorldfromTF();

      //Trajectory *traj = std::shared_ptr<TrajectoryComposite>(new TrajectoryComposite());
      TrajectoryFrames frames;

      std::vector<FeatureVector> features;

      // create KDL trajectory and compute frames along it
      {
        clock_t begin = clock();
        RotationalInterpolation_SingleAxis *ri = new RotationalInterpolation_SingleAxis();
        Trajectory_Composite *traj = new Trajectory_Composite();
        Path_RoundedComposite *path = new Path_RoundedComposite(0.01,0.02,ri);
        // construct a path
        for (double z = 0; z < 2.0; z += 0.4) {
          Rotation r1 = Rotation::RPY(0,0,0);
          Vector v1 = Vector(0,(z)*cos(z),z);
          Frame t1 = Frame(r1,v1);
          path->Add(t1);
        }
        path->Finish();

        double t = 0;
        double dt = 0.05;

		    VelocityProfile_Spline *velprof = new VelocityProfile_Spline();
        //VelocityProfile *velprof = new VelocityProfile_Trap(0.5,0.1);
		    velprof->SetProfile(0,path->PathLength());
        velprof->SetProfileDuration(0.0, 0.5, 3.0);
        velprof->Write(std::cout); std::cout << std::endl;

        Trajectory_Segment *seg = new Trajectory_Segment(path, velprof);

        traj->Add(seg);

        std::cout<<traj->Duration()<<std::endl;

        for (; t < traj->Duration(); t += dt) {
          Frame f = test.lookup(AGENT) * traj->Pos(t);
          frames.push_back(f);
        }

        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Computing trajectory of length " << frames.size() << " took " << elapsed_secs << " seconds." << std::endl;

        // convert to PoseArray for outputting debug trajectory
        pub.publish(toPoseArray(traj,0.05,"wam/wrist_palm_link"));

        delete traj;
      }

      // look at the time it takes to compute features
      {
        using namespace std;

        std::vector<std::string> test_set;
        //test_set.push_back("link");
        test_set.push_back("node");
        clock_t begin = clock();
        test.getFeaturesForTrajectory(features,test_set,frames);
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Computing features for " << features.size() << " positions took " << elapsed_secs << " seconds." << std::endl;
      }

      geometry_msgs::PoseArray msg;
      //msg.header.frame_id = "gbeam_link_1/gbeam_link"; //"wam/wrist_palm_link";
      msg.header.frame_id = "gbeam_node_1/gbeam_node"; //"wam/wrist_palm_node";

      //std::cout << "Feature size: " << features[0].size() << std::endl;

      for (int idx = 0; idx < features.size(); ++idx) {

        std::cout << "[" << idx << " ] Feature values: ";

        for (int i = 0; i < features[idx].size(); ++i) {
          std::cout << features[idx][i] << " ";
        }

        std::cout << std::endl;

        if (features[idx].size() < POSE_FEATURES_SIZE) continue;

        grid::Pose featureFrame;
        Features::featuresToPose(features[idx],featureFrame,0);

        geometry_msgs::Pose p;
        tf::Pose tfp;
        tf::poseKDLToTF(featureFrame,tfp);
        tf::poseTFToMsg(tfp,p);
        msg.poses.push_back(p);
      }

      fpub.publish(msg);

      rate.sleep();
    }
  } catch (ros::Exception ex) {
    ROS_ERROR("%s",ex.what());
  }
}
