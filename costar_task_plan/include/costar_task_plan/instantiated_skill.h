#ifndef _GRID_INSTANTIATED_SKILL
#define _GRID_INSTANTIATED_SKILL

#include <memory>

#include <costar_task_plan/test_features.h>
#include <costar_task_plan/dmp_trajectory_distribution.h>
#include <costar_task_plan/trajectory_distribution.h>
#include <costar_task_plan/robot_kinematics.h>
#include <costar_task_plan/skill.h>
#include <costar_task_plan/utils/params.h>


#include <actionlib/client/simple_action_client.h>
#include <costar_plan_msgs/CommandAction.h>

#include <random>

namespace costar {

  typedef std::shared_ptr<TestFeatures> TestFeaturesPtr;

  /**
   * creating predicates
   * What is going to change after we do this
   */
  struct PredicateEffect {
    std::string predicate;
    bool value;
    SkillPtr skill; // this is where we actually need to learn the effects model
  };

  class InstantiatedSkill;
  typedef std::shared_ptr<InstantiatedSkill> InstantiatedSkillPtr;

  /**
   * Defines a particular instance of a skill
   */
  class InstantiatedSkill {

  protected:

    static unsigned int next_id;

    const unsigned int id; // unique id for this skill
    bool done; // set to true if we don't need to keep evaluating this
    bool touched; // has anyone done anything with this skill yet

    std::unordered_map<std::string,std::string> assignment;
    SkillPtr skill; // the skill itself
    TrajectoryDistributionPtr spline_dist; // the path we end up taking for this skill

    std::vector<double> T; // probability of going to each of the possible next actions
    std::vector<double> last_T; // probability of going to each of the possible next actions
    std::vector<InstantiatedSkillPtr> next;

    // selected endpoints for this trajectory
    std::vector<JointTrajectoryPoint> end_pts;

    // store start points for this trajectory
    std::vector<JointTrajectoryPoint> start_pts;
    std::vector<double> start_ps;
    std::vector<unsigned int> prev_idx;

    std::vector<double> avg_next_ps;
    std::vector<std::vector<double> > next_ps;
    //std::vector<std::vector<unsigned int> > next_counts;
    std::vector<unsigned int> prev_counts;

    std::vector<double> prev_p_sums;
    std::vector<double> acc;


    std::vector<PredicateEffect> effects;

    RobotKinematicsPtr robot;

    Params p;

    // parameters
    double model_norm;
    double best_p;
    double transitions_step;
    unsigned int best_idx;
    unsigned int cur_iter;
    unsigned int good_iter;

    std::vector<FeatureVector> traj_features;

    Pose currentAttachedObjectFrame;

    //static std::uniform_real_distribution<double> unif_rand(0.,1.);
    //static std::default_random_engine re;

  public:

    TestFeaturesPtr features;
    DmpTrajectoryDistributionPtr dmp_dist; // the path we end up taking for this skill

    ros::Publisher *pub;

    // data
    std::vector<FeatureVector> params;
    std::vector<JointTrajectory> trajs;
    std::vector<double> ps;
    std::vector<double> my_ps;
    std::vector<double> my_future_ps;
    std::vector<double> iter_lls;
    std::vector<unsigned int> next_skill;
    
    double prior;
    double last_probability;
    unsigned int last_samples;
    bool useCurrentFeatures;


    void updateCurrentAttachedObjectFrame();
    const Pose &getAttachedObjectFrame() const;

    /**
     * normalize the transition probabilities
     */
    void updateTransitions();

    /** 
     * find best entries
     */
    void updateBest(unsigned int nsamples);

    /** 
     * default constructor
     */
    InstantiatedSkill();

    /**
     * set up with parameters
     */
    InstantiatedSkill(Params &p_);

    /**
     * set all variables back to original values
     * set children to not done too
     */
    void reset();

    /**
     * print out debug info on child probabilities and current ("my") probabilities
     */
    void debugPrintCurrentChildProbabilities(unsigned int samples);

    /**
     * create a new skill with dmps
     */
    static InstantiatedSkillPtr DmpInstance(SkillPtr skill,
                                                TestFeaturesPtr features,
                                                RobotKinematicsPtr robot,
                                                unsigned int nbasis,
                                                GridPlanner *checker = 0);

    /**
     * create a new skill with dmps
     */
    static InstantiatedSkillPtr DmpInstance(SkillPtr skill,
                                                SkillPtr grasp,
                                                TestFeaturesPtr features,
                                                RobotKinematicsPtr robot,
                                                unsigned int nbasis,
                                                GridPlanner *checker = 0);


    /**
     * create a new skill with spline and segments
     */
    static InstantiatedSkillPtr SplineInstance(SkillPtr skill,
                                                   TestFeaturesPtr features,
                                                   RobotKinematicsPtr robot,
                                                   unsigned int nseg);


    /*
     * set prior
     */
    InstantiatedSkill &setPrior(const double &prior);

    /**
     * create an empty root node
     */
    static InstantiatedSkillPtr Root();

    /**
     * define a possible child
     */
    InstantiatedSkill &addNext(InstantiatedSkillPtr skill);

    /**
     * run a single iteration of the loop. return a set of trajectories.
     * this is very similar to code in the demo
     * PROBABILITY is p(next_skill)
     * PS_OUT is the adjusted probability of each trajectory (normalized)
     */
    void step(const std::vector<double> &ps,
              const std::vector<trajectory_msgs::JointTrajectoryPoint> &start_pts,
              std::vector<double> &ps_out,
              //std::vector<unsigned int> &counts_out,
              double &probability,
              unsigned int len, // number of input samples provided (AKA prev samples)
              int horizon,
              unsigned int samples);

    /**
     * add some noise and refresh norm terms
     */
    void refresh(int horizon);

    /**
     * descend through the tree
     * execute as we reach nodes that require it
     * use gripper tool to send messages
     */
    bool execute(GridPlanner &gp,
                 actionlib::SimpleActionClient<costar_plan_msgs::CommandAction> &ac,
                 int horizon,
                 bool replan = false,
                 int replan_depth = 0);


    // randomly sample an index from the probabilities
    unsigned int sampleIndex(unsigned int nsamples) const;

    // initialize probabilities (or really any array of doubles)
    void initializePs(std::vector<double> &ps, double val);

    // initialize next probabilities
    void initializeNextPs(std::vector<std::vector<double> > &ps, double val);
    void initializeCounts(std::vector<unsigned int> &ps, unsigned int val);
    void accumulateProbs(const std::vector<double> &prev_ps, unsigned int len);
    void copyEndPoints(const std::vector<JointTrajectoryPoint> &prev_end_pts,
                       const std::vector<double> &prev_ps, unsigned int len);

    void publish();

  };


}

//#define LOW_PROBABILITY 0
//#define MAX_PROBABILITY 1
#define LOW_PROBABILITY -999999
#define MAX_PROBABILITY 0

#endif
