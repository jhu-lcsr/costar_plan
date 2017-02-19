#ifndef _GRID_TRAJECTORY_DISTRIBUTION
#define _GRID_TRAJECTORY_DISTRIBUTION

#include <costar/features.h>
#include <costar/test_features.h>
#include <costar/skill.h>
#include <costar/dist/gmm.h>

namespace costar {
  static const unsigned int SPLINE_POS1(0);
  static const unsigned int SPLINE_VEL1(1);
  static const unsigned int SPLINE_ACC1(2);
  static const unsigned int SPLINE_POS2(3);
  static const unsigned int SPLINE_VEL2(4);
  static const unsigned int SPLINE_ACC2(5);
  static const unsigned int SEGMENT_DURATION(6);
  static const unsigned int SPLINE_DIM(7);

  class TrajectoryDistribution {
  protected:
    gcop::Gmm<> dist; // stores distributions
    unsigned int nseg; // number of segments
    unsigned int nvars; // dimensionality of the trajectory space
    Pose initial;
    bool verbose;
    double diagonal_sigma;
    double def_step_size;

  public:

    /**
     * Initialize a trajectory distribution with velocity profile, etc.
     */
    TrajectoryDistribution(int nseg, int k = 1);

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
    void sample(std::vector<EigenVectornd> &params,std::vector<Trajectory *> &trajs);

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

  };

  typedef std::shared_ptr<TrajectoryDistribution> TrajectoryDistributionPtr;

}

#endif
