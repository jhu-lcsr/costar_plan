#include <costar_task_plan/instantiated_skill.h>

namespace costar {

  unsigned int InstantiatedSkill::next_id(0);

  /** 
   * default constructor
   */
  InstantiatedSkill::InstantiatedSkill()
    : id(next_id++), done(false), useCurrentFeatures(false),
    touched(false), spline_dist(0), dmp_dist(0), skill(0),
    trajs(), effects(), cur_iter(0), last_probability(MAX_PROBABILITY), last_samples(1u), pub(0), prior(0)
  {
  }

  /**
   * set up with parameters
   */
  InstantiatedSkill::InstantiatedSkill(Params &p_) :
    p(p_),
    id(next_id++), done(false), touched(false), spline_dist(0), dmp_dist(0), skill(0),
    effects(),
    ps(p_.ntrajs), iter_lls(p_.iter),
    useCurrentFeatures(false),
    trajs(p_.ntrajs),
    next_ps(),//p_.ntrajs),
    params(p_.ntrajs), cur_iter(0), good_iter(0),
    next_skill(p_.ntrajs),
    prev_idx(p_.ntrajs),
    end_pts(p_.ntrajs),
    start_pts(p_.ntrajs),
    start_ps(p_.ntrajs),
    my_ps(p_.ntrajs),
    my_future_ps(p_.ntrajs),
    prev_p_sums(p_.ntrajs),
    avg_next_ps(p_.ntrajs),
    //next_counts(),//p_.ntrajs),
    prev_counts(p_.ntrajs),
    transitions_step(p_.step_size),
    acc(p_.ntrajs),
    last_probability(MAX_PROBABILITY),
    last_samples(1u), pub(0), prior(0)
  {
    done = false;
    touched = false;
    model_norm = p.base_model_norm;
    best_p = LOW_PROBABILITY;
    cur_iter = 0;
    good_iter = 0;
    best_idx = 0;
  }

  /**
   * create a new skill with dmps
   */
  InstantiatedSkillPtr InstantiatedSkill::DmpInstance(SkillPtr skill,
                                                      TestFeaturesPtr features,
                                                      RobotKinematicsPtr robot,
                                                      unsigned int nbasis,
                                                      CostarPlanner *checker)
  {

    Params p = readRosParams();
    InstantiatedSkillPtr is(new InstantiatedSkill(p));
    is->skill = skill;
    is->features = features;
    is->robot = robot;
    is->dmp_dist = DmpTrajectoryDistributionPtr(
        new DmpTrajectoryDistribution(robot->getDegreesOfFreedom(),
                                      nbasis,
                                      robot));
    is->dmp_dist->attachObjectFrame(skill->getDefaultAttachedObjectPose());
    is->dmp_dist->initialize(*features,*skill);
    is->dmp_dist->setCollisionDetectionStep(p.collision_detection_step);

    if(checker) {
      is->dmp_dist->setCollisionChecker(checker);
    }

    return is;
  }

  /**
   * create a new skill with dmps
   */
  InstantiatedSkillPtr InstantiatedSkill::DmpInstance(SkillPtr skill,
                                                      SkillPtr grasp,
                                                      TestFeaturesPtr features,
                                                      RobotKinematicsPtr robot,
                                                      unsigned int nbasis,
                                                      CostarPlanner *checker)
  {

    Params p = readRosParams();
    InstantiatedSkillPtr is(new InstantiatedSkill(p));
    is->skill = skill;
    is->features = features;
    is->robot = robot;
    is->dmp_dist = DmpTrajectoryDistributionPtr(
        new DmpTrajectoryDistribution(robot->getDegreesOfFreedom(),
                                      nbasis,
                                      robot));
    is->dmp_dist->attachObjectFromSkill(*grasp);
    is->dmp_dist->initialize(*features,*skill);

    if(checker) {
      is->dmp_dist->setCollisionChecker(checker);
    }

    return is;
  }

  // initialize probabilities (or really any array of doubles)
  void InstantiatedSkill::initializePs(std::vector<double> &ps_, double val) {
      for (unsigned int i = 0; i < ps_.size(); ++i) {
        ps_[i] = val;
      }
  }

  // initialize next probabilities
  void InstantiatedSkill::initializeNextPs(std::vector<std::vector<double> > &ps_, double val) {
    for (unsigned int next = 0; next < ps_.size(); ++next) {
      for (unsigned int i = 0; i < ps_[next].size(); ++i) {
        ps_[next][i] = val;
      }
    }
  }

  /**
   * create a new skill with spline and segments
   */
  InstantiatedSkillPtr InstantiatedSkill::SplineInstance(SkillPtr skill,
                                                         TestFeaturesPtr features,
                                                         RobotKinematicsPtr robot,
                                                         unsigned int nseg)
  {

    InstantiatedSkillPtr is(new InstantiatedSkill());
    is->skill = skill;
    is->features = features;
    is->robot = robot;
    is->spline_dist = TrajectoryDistributionPtr(new TrajectoryDistribution(nseg));
    is->spline_dist->initialize(*features,*skill);

    return is;
  }

  /**
   * create an empty root node
   */
  InstantiatedSkillPtr InstantiatedSkill::Root() {
    Params p = readRosParams();
    return InstantiatedSkillPtr (new InstantiatedSkill(p));
  }

  /**
   * define a possible child
   */
  InstantiatedSkill &InstantiatedSkill::addNext(InstantiatedSkillPtr skill) {
    next.push_back(skill);
    T.push_back(1);
    last_T.push_back(1);
    next_ps.resize(T.size());
    //next_counts.resize(T.size());

    for (auto &next_ps_vec: next_ps) {
      next_ps_vec.resize(p.ntrajs);
    }

    //for (auto &next_count_vec: next_counts) {
    //  next_count_vec.resize(p.ntrajs);
    //}

    if (p.random_transitions and T.size() > 1) {
      unsigned int idx = rand() % T.size();
      std::cout << "(Test setup) RANDOM IDX = " << idx << "\n";
      for (unsigned int i = 0; i < T.size(); ++i) {
        if (i != idx) {
          T[i] = 0;
          last_T[i] = 0;
        } else {
          T[i] = 1;
          last_T[i] = 1;
        }
      }
    }

    for(unsigned int i = 0; i < T.size(); ++i) {
      T[i] = 1;
      last_T[i] = 1;
    }
    updateTransitions();
  }

  void InstantiatedSkill::initializeCounts(std::vector<unsigned int> &ps, unsigned int val) {
    for (auto &u: ps) {
      u = val;
    }
  }

}
