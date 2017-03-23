#include <costar_task_plan/utils/params.h>
#include <random>
#include <ctime>

namespace costar {

    Params readRosParams() {
      Params p;
      ros::NodeHandle nh_tilde("~");
      if (not nh_tilde.getParam("step_size",p.step_size)) {
        p.step_size = 0.80;
      }
      if (not nh_tilde.getParam("noise",p.noise)) {
        p.noise = 1e-10;
      }
      if (not nh_tilde.getParam("ntrajs",p.ntrajs)) {
        p.ntrajs = 50;
      }
      if (not nh_tilde.getParam("iter",p.iter)) {
        p.iter = 10;
      }
      if (not nh_tilde.getParam("skill",p.skill_name)) {
        p.skill_name = "approach";
      }
      if (not nh_tilde.getParam("goal",p.goal_name)) {
        p.goal_name = "grasp";
      }
      if (not nh_tilde.getParam("base_model_norm",p.base_model_norm)) {
        p.base_model_norm = 0.01;
      }
      if (not nh_tilde.getParam("model_norm_step",p.model_norm_step)) {
        //p.model_norm_step = 0.1;
        p.model_norm_step = 1.0;
      }
      if (not nh_tilde.getParam("base_sampling_noise",p.base_sampling_noise)) {
        p.base_sampling_noise = 0.01;
      }
      if (not nh_tilde.getParam("sampling_noise_step",p.sampling_noise_step)) {
        p.sampling_noise_step = 0.1;
      }
      if (not nh_tilde.getParam("wait",p.wait)) {
        p.wait = 0.25;
      }
      if (not nh_tilde.getParam("update_horizon",p.update_horizon)) {
        p.update_horizon = 1e-2;
      }
      if (not nh_tilde.getParam("detect_collisions",p.detect_collisions)) {
        p.detect_collisions = false;
      }
      if (not nh_tilde.getParam("verbosity",p.verbosity)) {
        p.verbosity = 0;
      }
      if (not nh_tilde.getParam("starting_horizon",p.starting_horizon)) {
        p.starting_horizon = 2;
      }
      if (not nh_tilde.getParam("max_horizon",p.max_horizon)) {
        p.max_horizon = 5;
      }
      if (not nh_tilde.getParam("collisions_verbose",p.collisions_verbose)) {
        p.collisions_verbose = false;
      }
      if (not nh_tilde.getParam("test",p.test)) {
        p.test = 0;
      }
      if (not nh_tilde.getParam("compute_statistics",p.compute_statistics)) {
        p.compute_statistics = true;
      }
      if (not nh_tilde.getParam("distribution_noise",p.distribution_noise)) {
        p.distribution_noise = 1e-8;
      }
      if (not nh_tilde.getParam("collision_detection_step",p.collision_detection_step)) {
        p.collision_detection_step = 4;
      }
      if (not nh_tilde.getParam("replan_depth",p.replan_depth)) {
        p.replan_depth = 4;
      }
      if (not nh_tilde.getParam("execute_depth",p.execute_depth)) {
        p.execute_depth = p.max_horizon;
      }
      if (not nh_tilde.getParam("fixed_distribution_noise",p.fixed_distribution_noise)) {
        p.fixed_distribution_noise = true;
      }
      if (not nh_tilde.getParam("random_transitions",p.random_transitions)) {
        p.random_transitions = false;
      }
      if (not nh_tilde.getParam("randomize",p.randomize)) {
        p.randomize = false;
      } else if (p.randomize) {
        time_t timer;
        time(&timer);
        srand(timer);
      }

      return p;
    }


  }
