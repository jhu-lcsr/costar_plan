#ifndef _GRID_FLOAT_FEATURES
#define _GRID_FLOAT_FEATURES

#include <costar_task_plan/features.h>

// training features reads the feature data from ROS topics
#include <ros/ros.h>

namespace costar {
  
  class TrainingFeatures: public Features {

    public:

    /* getPose
     * This function needs to be implemented by inheriting classes.
     * Time field helps determine when the query should occur.
     * A feature query gets the set of all featutes for different points in time, normalizes them, and returns.
     */
    std::vector<Pose> getPose(const std::string &name,
                                unsigned long int mintime = 0,
                                unsigned long int maxtime = 0);

    /* getFeatureValues
     * Returns a list of features converted into a format we can use.
     */
    std::vector<std::vector<double> > getFeatureValues(const std::string &name,
                                                          unsigned long int mintime = 0,
                                                          unsigned int long int maxtime = 0);
    /**
     * open
     * Open a rosbag containing the demonstrations.
     * We make some assumptions as to how these are stored.
     */
    void open(const std::string &bagfile);

    protected:
    rosbag::Bag bag; // current bag holding all demonstration examples
  };

}

#endif
