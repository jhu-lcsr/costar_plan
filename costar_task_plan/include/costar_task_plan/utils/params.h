#ifndef _GRID_PARAMS
#define _GRID_PARAMS

#include <ros/ros.h>

namespace grid {


  struct Params {
    double base_model_norm;
    double model_norm_step;
    double base_sampling_noise;
    double sampling_noise_step;
    double step_size;
    double noise; // initial distribution noise
    double wait;
    int ntrajs; // number of trajectories to sample
    int iter;
    int starting_horizon; // how far ahead are we looking to start with
    int max_horizon; // how far ahead are we allowed to look when planning
    int verbosity;
    bool detect_collisions;
    std::string skill_name;
    std::string goal_name;
    double update_horizon; // when do we increase the horizon?
    double distribution_noise; // fixed noise to add to distribution at each iteration
    bool collisions_verbose; // should we output lots of info from the collision checker
    bool compute_statistics; // should we send statistics messages
    int test; // which test are we running
    int collision_detection_step; // how many trajectory points to skip when doing collision detection
    int replan_depth; // how deep can we go into the tree when replanning? 0 = max_horizon
    int execute_depth; // how deep can we go into the tree when executing? 0 = max_horizon
    bool fixed_distribution_noise;
    bool random_transitions;
    bool randomize;
  };

  Params readRosParams();


}

#endif
