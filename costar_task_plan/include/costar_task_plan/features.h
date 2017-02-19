#ifndef _GRID_FEATURES
#define _GRID_FEATURES

#include <vector>
#include <unordered_map>
#include <memory>

//#include <gcop/pose.h>

//#include <Eigen/Geometry>
#include <kdl/frames.hpp>
#include <kdl/trajectory.hpp>
#include <tf_conversions/tf_kdl.h>

#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>
#include <costar/dist/utils.h>

#include <costar/robot_kinematics.h>

/**
 * Features
 * Parent abstract class for a set of robot features.
 * Abstract method produces appropriate poses given queries.
 */

namespace costar {

  // feature distribution defined here as a GCOP gmm for now

  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> EigenVectornd;

  // use array of poses for now (not using velocity/commands)
  typedef std::vector<Pose> TrajectoryFrames;

  // trajectories are represented as KDL trajectories
  typedef KDL::Trajectory Trajectory;

  //typedef std::vector<double> FeatureVector;
  typedef EigenVectornd FeatureVector;

  std::vector<double> poseToArray(const Pose &pose);

  /* FeatureType
   * Save each feature as its own thing.
   */
  typedef enum FeatureType { POSE_FEATURE, FLOAT_FEATURE, TIME_FEATURE } FeatureType;

  static const std::string AGENT("agent");
  static const std::string BASE("base");
  static const unsigned int POSE_FEATURE_X(0);
  static const unsigned int POSE_FEATURE_Y(1);
  static const unsigned int POSE_FEATURE_Z(2);


  /* index definitions for creating trajectory poses */
  static const unsigned int POSE_RPY_SIZE(6);
  static const unsigned int POSE_FEATURE_ROLL(3);
  static const unsigned int POSE_FEATURE_PITCH(4);
  static const unsigned int POSE_FEATURE_YAW(5);

  static const unsigned int POSE_FEATURES_SIZE_ND(8);
  static const unsigned int POSE_FEATURES_SIZE(12);
  static const unsigned int POSE_FEATURE_WX(3);
  static const unsigned int POSE_FEATURE_WY(4);
  static const unsigned int POSE_FEATURE_WZ(5);
  static const unsigned int POSE_FEATURE_WW(6);
  static const unsigned int POSE_FEATURE_DIST(7);
  static const unsigned int POSE_FEATURE_DX(8);
  static const unsigned int POSE_FEATURE_DY(9);
  static const unsigned int POSE_FEATURE_DZ(10);
  static const unsigned int POSE_FEATURE_DDIST(11);

  //static const Pose rotationHack(KDL::Rotation::RotY(M_PI/2));
  static const Pose rotationHack(KDL::Rotation::RotY(0));

  static const unsigned int FLOAT_FEATURES_SIZE(1);
  static const unsigned int TIME_FEATURES_SIZE(1);

  static const std::string JOINT_STATES_TOPIC("joint_states");
  static const std::string GRIPPER_MSG_TOPIC("gripper_msg");
  static const std::string BASE_TFORM_TOPIC("base_tform");
  static const std::string GRIPPER_CMD("gripper_cmd");

  class Features {
  public:

    /**
     * Do we need to use diff features to train this?
     */
    void setUseDiff(const bool use_diff);

    /**
     * set all values in the vector to something
     */
    void setAll(std::vector<FeatureVector> &features,
                const std::vector<std::string> &all_features,
                const std::string &name,
                const double &value);

    /* getPose
     * This function needs to be implemented by inheriting classes.
     * Time field helps determine when the query should occur.
     * A feature query gets the set of all featutes for different points in time, normalizes them, and returns.
     */
    virtual TrajectoryFrames getPose(const std::string &name,
                                     double mintime = 0,
                                     double maxtime = 0) = 0;

    /* getFeatureValues
     * Returns a list of features converted into a format we can use.
     */
    virtual std::vector< FeatureVector > getFeatureValues(const std::string &name,
                                                          double mintime = 0,
                                                          double maxtime = 0) = 0;

    /*
     * Processing code.
     * Needs to, among other things, get all the features and turn them into normalized data.
     */
    void initialize();

    /**
     * Get a structure containing all the appropriate features
     */
    std::vector< FeatureVector > getFeatures(std::vector<std::string> &names);

    /**
     * add a feature
     * also updates expected array size
     */
    Features &addFeature(const std::string &name, const FeatureType type, unsigned int size = 0);

    /**
     * get the type of a feature
     */
    FeatureType getFeatureType(const std::string &name) const;

    /**
     * getFeaturesSize
     * Compute number of features we expect to see
     */
    unsigned int getFeaturesSize() const;

    /**
     * getFeaturesSize
     * Compute number of features we expect to see
     */
    unsigned int getFeaturesSize(const std::string &name) const;

    /**
     * getFeaturesSize
     * Compute number of features we expect to see
     */
    unsigned int getFeaturesSize(const std::vector<std::string> &names) const;

    /**
     * getFeaturesSize
     * Compute number of features we expect to see
     */
    unsigned int getFeaturesSize(const std::vector<std::string> &names, bool use_diff) const;


    /**
     * getFeaturesSizeNoDiff
     */
    unsigned int getFeaturesSizeNoDiff(const std::vector<std::string> &names) const;

    /**
     * getPoseFeatures
     * Load pose data at index
     */
    static void getPoseFeatures(const Pose &pose, FeatureVector &f, unsigned int idx, bool use_diff = true);

    /**
     * getPoseFeatures
     * Load pose data at index
     */
    static void getPoseFeatures(const Pose &pose, FeatureVector &f, unsigned int idx, FeatureVector &prev, bool use_diff = true);

    /**
     * featuresToPose
     */
    static void featuresToPose(FeatureVector &f, Pose &p, unsigned int idx);

    /** 
     * setRobotKinematics
     * sets the thing that will actually compute forward and inverse kinematics for our robot
     */
    void setRobotKinematics(std::shared_ptr<RobotKinematics> rk);

    /**
     * updateFeaturesSize
     * Compute number of features we expect to see
     */
    void updateFeaturesSize();

    /**
     * attach an object frame to this set of features by identifier
     */
    Features &attachObjectFrame(const std::string &object);

    /**
     * detach all object frames
     */
    Features &detachObjectFrame();

    /**
     * return the name of an attached object
     */
    const std::string &getAttachedObject() const;

    /**
     * is there an object here
     */
    bool hasAttachedObjectFrame() const;


    /** 
     * initializers
     */
    Features();

  protected:

    bool use_diff;

    bool attached;
    Pose attachedObjectFrame;
    std::string attachedObject;

    unsigned int features_size;
    unsigned int features_size_nd;
    std::unordered_map<std::string,FeatureType> feature_types;
    std::unordered_map<std::string,unsigned int> feature_sizes;
    std::unordered_map<std::string,unsigned int> feature_sizes_nd;

    RobotKinematicsPtr robot; // stores the robot itself
    unsigned int n_dof;
  };
}

#endif
