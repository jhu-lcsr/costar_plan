#ifndef _GRID_TEST_FEATURES
#define _GRID_TEST_FEATURES

#include <costar_task_plan/features.h>
#include <tf/transform_listener.h>

namespace costar {

  class TestFeatures : public Features {

  public:

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
                                                double maxtime = 1);

    /* getFeaturesForTrajectory
     * Get information for a single feature over the whole trajectory given in traj.
     * Traj is KDL::Trajectory
     */
    void getFeaturesForTrajectory(std::vector<FeatureVector> &features,
                                                        const std::vector<std::string> &names,
                                                        Trajectory *traj,
                                                        double dt = 0.05);

    /* getFeaturesForTrajectory
     * Get information for a single feature over the whole trajectory given in traj.
     * Traj is a set of frames
     */
    void getFeaturesForTrajectory(std::vector<FeatureVector> &features,
        const std::vector<std::string> &name,
        const TrajectoryFrames &traj);

    /* getFeaturesForTrajectory
     * Get information for a single feature over the whole trajectory given in traj.
     * Traj is a set of frames
     * Uses an attached object frame
     */
    void getFeaturesForTrajectory(std::vector<FeatureVector> &features,
                                  const std::vector<std::string> &name,
                                  const TrajectoryFrames &traj,
                                  const bool useAttachedObjectFrame,
                                  const Pose &attachedObjectFrame);

    /* addFrame
     * Adds a frame of reference as a feature
     */
    TestFeatures &setFrame(const std::string &frame, const std::string &objectClass);

    /* addAgent:
     * configure agent's manipulation frame
     */
    TestFeatures &setAgentFrame(const std::string &agentFrame);

    /* addBase:
     * configure base's manipulation frame
     */
    TestFeatures &setBaseFrame(const std::string &baseFrame_);

    /* configure world frame for this TestFeatures object
    */
    TestFeatures &setWorldFrame(const std::string &worldFrame);

    /**
     * get the world frame
     */
    const std::string &getWorldFrame() const;

    /* lookup tf frame for key
     * in world frame
     */
    Pose lookup(const std::string &key);

    /* lookup tf frame for key
    */
    Pose lookup(const std::string &key, const std::string &in_frame);

    /*
     * run lookup for all objects
     * store results for poses from tf
     */
    TestFeatures &updateWorldfromTF();

    /**
     * get the current end effector position
     */
    Pose getCurrentEndEffector() const;

    /**
     * get current attached object frame
     */
    const Pose &getAttachedObjectFrame() const;

    std::vector<std::string> getClasses() const;

    /**
     * get all the current IDs (coordinate frames)
     */
    std::vector<std::string> getIds() const;

  private:
    std::unordered_map<std::string, std::string> objectClassToID;
    std::unordered_map<std::string, Pose> currentPose; // used for fast lookup
    std::unordered_map<std::string, Pose> currentPoseInv; // used for fast lookup

    // name of the frame representing the end effector, i.e. what are we planning from?
    std::string agentFrame;

    // name of the base frame
    std::string baseFrame;

    // name of the root/world frame
    std::string worldFrame;

    // tf
    tf::TransformListener listener;
  };
}

#endif
