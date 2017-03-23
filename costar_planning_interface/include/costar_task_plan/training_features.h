#ifndef _GRID_TRAINING_FEATURES
#define _GRID_TRAINING_FEATURES

#include <costar_task_plan/features.h>

#include <unordered_map>

// training features reads the feature data from ROS topics
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/JointState.h>

#include <memory>

namespace costar {

  struct WorldConfiguration {
    ros::Time t; 
    std::unordered_map<std::string,Pose> object_poses;
    Pose base_tform;
    sensor_msgs::JointState joint_states;
    Pose ee_tform;
    std::vector<double> gripper_cmd;
  };

  class TrainingFeatures: public Features {

  public:


    /**
     * get last pose
     * use this TrainingFeatures to lookup the object you need
     */
    Pose getPoseFrom(const std::string &name, FeatureVector f);

    /* getPose
     * This function needs to be implemented by inheriting classes.
     * Time field helps determine when the query should occur.
     * A feature query gets the set of all featutes for different points in time, normalizes them, and returns.
     */
    std::vector<Pose> getPose(const std::string &name,
                              double mintime = 0,
                              double maxtime = 0);

    /* getFeatureValues
     * Returns a list of features converted into a format we can use.
     */
    std::vector<FeatureVector> getFeatureValues(const std::string &name,
                                                double mintime = 0,
                                                double maxtime = 0);

    /**
     * getFeatureValues
     * get all available features from provided set
     */
    std::vector<FeatureVector> getFeatureValues(const std::vector<std::string> &features);

    /**
     * get all available features
     * for testing, at least for now
     */
    std::vector<FeatureVector> getAllFeatureValues();

    /**
     * helper
     * convert a world into a set of features
     */
    FeatureVector worldToFeatures(const WorldConfiguration &w) const;

    /**
     * helper
     * convert a world into a set of features
     */
    FeatureVector worldToFeatures(const WorldConfiguration &w, const std::vector<std::string> &features) const;

    /**
     * helper
     * convert a world into a set of features
     */
    FeatureVector worldToFeatures(const WorldConfiguration &w, FeatureVector &prev) const;

    /**
     * helper
     * convert a world into a set of features
     */
    FeatureVector worldToFeatures(const WorldConfiguration &w, const std::vector<std::string> &features, FeatureVector &prev) const;
    /**
     * read
     * Open a rosbag containing the demonstrations.
     * We make some assumptions as to how these are stored.
     * This function will read in the poses and other information associated with the robot.
     * This information all gets stored and can be used to compute features or retrieve world configurations.
     */
    void read(const std::string &bagfile, int downsample = 0);

    /**
     * initialize training features with the necessary world objects to find
     */
    TrainingFeatures(const std::vector<std::string> &objects);

    /**
     * print basic info for debugging
     */
    void printTrainingFeaturesInfo();

    /**
     * print all extractable features for the different objects
     */
    void printExtractedFeatures();

    const std::vector<WorldConfiguration> &getData() const;

  protected:

    ros::Time min_t;
    ros::Time max_t;

    /*
     * return the gripper features from a rosbag
     * must be implemented for the specific gripper being used
     */
    virtual std::vector<double> getGripperFeatures(rosbag::MessageInstance const &m) = 0;


    std::vector<std::string> objects; // objects we need
    std::vector<std::string> topics; // topics to pull from rosbag
    rosbag::Bag bag; // current bag holding all demonstration examples
    std::vector<WorldConfiguration> data; //all loaded data
    std::unordered_map<std::string,std::string> topic_to_object; //maps topics back to objects

    /**
     * return the joint states data we care about
     */
    Pose getJointStatesData(rosbag::MessageInstance const &m);

    /**
     * get the object poses we care about
     */
    Pose getObjectData(rosbag::MessageInstance const &m);

  };

}

#endif
