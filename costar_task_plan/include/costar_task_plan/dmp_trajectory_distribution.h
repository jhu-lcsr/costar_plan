#ifndef _GRID_DMP_TRAJECTORY_DISTRIBUTION
#define _GRID_DMP_TRAJECTORY_DISTRIBUTION

#include <memory>

#include <costar_task_plan/features.h>
#include <costar_task_plan/test_features.h>
#include <costar_task_plan/skill.h>
#include <costar_task_plan/dist/gmm.h>
#include <costar_task_plan/robot_kinematics.h>
#include <costar_task_plan/costar_planner.h>

#include <kdl/jntarray.hpp>

#include <trajectory_msgs/JointTrajectory.h>

// primitives for motion planning
#include <dmp/dmp.h>


using trajectory_msgs::JointTrajectory;
using trajectory_msgs::JointTrajectoryPoint;

namespace costar {

  typedef std::shared_ptr<GridPlanner> TrajectoryCheckerPtr;

  /**
   * Ideally this class would inherit from the same parent as TrajectoryDistribution
   * For now, though, I'm doing this separately to expedite things.
   *
   * Parameterize the class basically the same way:
   * - provide a Skill
   * - provide a RobotKinematics
   */
  class DmpTrajectoryDistribution {

  public:

    /**
     * Initialize a trajectory distribution with given params
     */
    DmpTrajectoryDistribution(unsigned int dim, unsigned int nbasis, RobotKinematicsPtr robot);

    /**
     * initialize
     * set up the distribution based on a skill and an environment
     */
    void initialize(TestFeatures &features, const Skill &skill, bool initBegin = false, std::vector<double> sigma = std::vector<double>());

    /**
     * initialize pose
     * set up the distribution based on a skill and an environment
     */
    void initializePose(TestFeatures &features, const Skill &skill, bool initBegin = false);

    /**
     * sample
     * Pull a random trajectory from the gmm
     * Convert it into a KDL trajectory
     * NON-CONST becuse Gmm::sample is likewise non-const
     */
    unsigned int sample(
        const std::vector<JointTrajectoryPoint> &start_pts,
        std::vector<EigenVectornd> &params,
        std::vector<JointTrajectory> &trajs,
        unsigned int nsamples = 0);

    /**
     * update
     * take a set of trajectories and samples
     * use the trajectories to reweight the distribution
     */
    void update(std::vector<EigenVectornd> &params,
                std::vector<double> &ps,
                unsigned int nsamples);


    /**
     * update
     * take a set of trajectories and samples
     * use the trajectories to reweight the distribution
     */
    void update(std::vector<EigenVectornd> &params,
                std::vector<double> &ps,
                unsigned int nsamples,
                double diagonal_noise);


    /**
     * update
     * take a set of trajectories and samples
     * use the trajectories to reweight the distribution
     */
    void update(std::vector<EigenVectornd> &params,
                std::vector<double> &ps,
                unsigned int nsamples,
                double diagonal_noise,
                double step_size);

    void addNoise(double d);
    void addNoise(std::vector<double> d);

  /**
   * use a model of a skill
   * compute the grasp pose from that skill and use it to plan for now
   */
  void attachObjectFromSkill(Skill &skill);

    /**
     * set an attached object
     */
    void attachObjectFrame(const Pose &pose);

    /*
     * remove object
     */
    void detachObjectFrame();

    /** is there an object attached */
    bool hasAttachedObject() const;

    /** get the attached object frame */
    const Pose &getAttachedObjectFrame() const;

    /** check for collisions */
    void setCollisionChecker(GridPlanner *);

    /**
     * set the skip between calling collision detection
     */
    unsigned int setCollisionDetectionStep(unsigned int step);

  protected:
    gcop::Gmm<> dist; // stores distributions
    GridPlanner *checker;

    Pose initial;
    bool verbose;
    double diagonal_sigma;
    double def_step_size;

    RobotKinematicsPtr robot; // used to get the joint states for this dmp

    unsigned int nbasis; // number of basis functions
    unsigned int dim; // number of dimensions

    unsigned int nvars;
    unsigned int collision_detection_step;

    std::vector<dmp::DMPData> dmp_list;
    std::vector<double> dmp_goal;

    double k_gain;
    double d_gain;
    double tau;
    std::vector<double> goal_threshold;

    KDL::JntArray q;

    double dmp_velocity_multiplier;


    // attached object
    bool attached;
    Pose attachedObjectFrame;

  };


  typedef std::shared_ptr<DmpTrajectoryDistribution> DmpTrajectoryDistributionPtr;
}

#endif
