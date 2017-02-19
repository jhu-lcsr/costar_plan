#ifndef _GRID_DMP_TRAJECTORY_DISTRIBUTION
#define _GRID_DMP_TRAJECTORY_DISTRIBUTION

#include <costar/features.h>
#include <costar/test_features.h>
#include <costar/skill.h>
#include <costar/dist/gmm.h>
#include <costar/robot_kinematics.h>

#include <kdl/jntarray.hpp>

#include <trajectory_msgs/JointTrajectory.h>

// primitives for motion planning
#include <dmp/dmp.h>


using trajectory_msgs::JointTrajectory;

namespace costar {


  /**
   * Ideally this class would inherit from the same parent as TrajectoryDistribution
   * For now, though, I'm doing this separately to expedite things.
   *
   * Parameterize the class basically the same way:
   * - provide a Skill
   * - provide a RobotKinematics
   */
  class CartDmpTrajectoryDistribution {

  public:

    /**
     * Initialize a trajectory distribution with given params
     */
    CartDmpTrajectoryDistribution(unsigned int dim, unsigned int nbasis, RobotKinematicsPointer robot);

    /**
     * initialize
     * set up the distribution based on a skill and an environment
     */
    void initialize(TestFeatures &features, const Skill &skill, std::vector<double> sigma = std::vector<double>());

    /**
     * sample
     * Pull a random trajectory from the gmm
     * Convert it into a KDL trajectory
     * NON-CONST becuse Gmm::sample is likewise non-const
     */
    void sample(std::vector<EigenVectornd> &params,std::vector<JointTrajectory> &trajs);

    /**
     * update
     * take a set of trajectories and samples
     * use the trajectories to reweight the distribution
     */
    void update(std::vector<EigenVectornd> &params,
                std::vector<double> &ps);


    /**
     * update
     * take a set of trajectories and samples
     * use the trajectories to reweight the distribution
     */
    void update(std::vector<EigenVectornd> &params,
                std::vector<double> &ps,
                double diagonal_noise);


    /**
     * update
     * take a set of trajectories and samples
     * use the trajectories to reweight the distribution
     */
    void update(std::vector<EigenVectornd> &params,
                std::vector<double> &ps,
                double diagonal_noise,
                double step_size);


  protected:
    gcop::Gmm<> dist; // stores distributions

    Pose initial;
    bool verbose;
    double diagonal_sigma;
    double def_step_size;

    RobotKinematicsPointer robot; // used to get the joint states for this dmp

    unsigned int nbasis; // number of basis functions
    unsigned int dim; // number of dimensions

    unsigned int nvars;

    std::vector<dmp::DMPData> dmp_list;
    std::vector<double> dmp_goal;

    double k_gain;
    double d_gain;
    double tau;
    std::vector<double> goal_threshold;

    KDL::JntArray q;
    
    double dmp_velocity_multiplier;

  };


}

#endif
