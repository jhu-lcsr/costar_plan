#include <costar_task_plan/skill.h>

namespace costar {

  /**
   * concatenate
   * Stick two actions together to generate a combined set of constraints.
   * This is an "OR" operation, not an "AND".
   */
  Skill Skill::concatenate(const Skill &a) const {
    std::cerr << __FILE__ << ":" << __LINE__ << ": Concatenate not implemented!" << std::endl;
  }

  /**
   * return a pose associated with object frame for a given feature
   */
  Pose &Skill::getInitializationFinalPose() {
    return init_final;
  }

  /**
   * return a pose associated with object frame for a given feature
   */
  Pose Skill::getInitializationFinalPose() const {
    return init_final;
  }

  /** return a default pose to plan motions with an attached object */
  const Pose &Skill::getDefaultAttachedObjectPose() const {
    return default_attached;
  }

  /**
   * return a pose associated with object frame for a given feature
   */
  Pose &Skill::getInitializationStartPose() {
    return init_start;
  }

  /**
   * return a pose associated with object frame for a given feature
   */
  Pose Skill::getInitializationStartPose() const {
    return init_start;
  }

  /**
   * make this a static skill (no movement)
   */
  Skill &Skill::setStatic(bool set_static) {
    is_static = set_static;

    if (is_static) {

    }

    return *this;
  }

  /**
   * is this static
   */
  bool Skill::isStatic() const {
    return is_static;
  }

  /**
   * default skill;
   * one object feature; identity covariance; k=1 gmm
   */
  Skill Skill::DefaultSkill(const std::string &name, const std::string &object) {

    //Skill skill(1,POSE_FEATURES_SIZE);
    Skill skill(name,1);
    skill.feature_names.push_back(object);
    skill.best_feature_name = object;

    return skill;
  }

  /**
   * return the exact list of features used for training this
   */
  const std::vector<std::string> &Skill::getFeatures() const {
    return feature_names;
  }

  /**
   * return the name of the best feature and only that feature
   */
  const std::string &Skill::getInitializationFeature() const {
    return best_feature_name;
  }

  /**
   * set the probability of seeing this skill at all
   */
  Skill &Skill::setPrior(const double &_prior) {
    prior = _prior;
    log_prior = log(_prior);
    return *this;
  }

  /**
   * create a skill based on a set of features and a number of clusters
   */
  Skill::Skill(int k, std::vector<std::string> &feature_names_, Features &features) :
    feature_names(feature_names_), attached_object(""), is_static(false), prior(1), log_prior(0)
  {
    unsigned int dim = features.getFeaturesSize();
    model = GmmPtr(new Gmm(dim,k));
  }

  /**
   * create a skill based on k and d
   */
  Skill::Skill(const std::string &name_, int k_) : name(name_), k(k_), attached_object(""), is_static(false), prior(1), log_prior(0) {
    // do nothing for now
  }

  /**
   * which feature should we use to initialize things?
   */
  Skill &Skill::setInitializationFeature(const std::string &feature) {
    best_feature_name = feature;
    return *this;
  }

  /**
   * add a feature to this skill
   */
  Skill &Skill::appendFeature(const std::string &feature) {
    feature_names.push_back(feature);
    if (best_feature_name.size() < 1) {
      best_feature_name = feature;
    }
    return *this;
  }

  /**
   * Adding training data
   * What data do we want to use? add as a Features object
   * Store this as a vector
   */
  void Skill::addTrainingData(TrainingFeatures &data) {

    data.setUseDiff(not isStatic());
    std::vector<FeatureVector> ex_data = data.getFeatureValues(feature_names);

    for (FeatureVector &ex: ex_data) {
      std::pair<FeatureVector,double> obs(ex,1.0);
      training_data.push_back(obs);
      if (isStatic()) {
        break;
      }
    }

    init_final = data.getPoseFrom(best_feature_name,*ex_data.rbegin());
    init_start = data.getPoseFrom(best_feature_name,*ex_data.begin());

    if (data.hasAttachedObjectFrame()) {
      std::cout << __FILE__ << ":" << __LINE__ << "\n";
      std::cout << "Attached object:" << data.getAttachedObject() << "\n";
      //default_attached = data.getPoseFrom(data.getAttachedObject(),*ex_data.begin());
      //std::cout << default_attached << "\n";

      //std::cout << "--------\n";

      WorldConfiguration w = *data.getData().begin();
      Pose obj = w.object_poses[data.getAttachedObject()];
      //Pose ee = w.ee_tform;

      //std::cout << obj << "\n" << ee << "\n" << obj.Inverse() * w.base_tform * w.ee_tform << "\n";
      default_attached = obj.Inverse() * w.base_tform * w.ee_tform;
    }
  }

  /**
   * trainSkillModel
   * train the model for expected features while learning this action
   */
  void Skill::trainSkillModel() {
    if (training_data.size() > 0 && training_data[0].first.size() > 0) {

      normalized_training_data.clear();

      unsigned int dim = training_data[0].first.size();
      //std::cout << "DIM = " << dim << "\n";
      mean = FeatureVector(dim);
      std = FeatureVector(dim);

      mean.setZero();
      std.setZero();

      // loop to compute mean
      double num_examples = (double)training_data.size();
      std::cout << "[" << getName() << "] Training from " << num_examples << " examples.\n";
      for (auto &pair: training_data) {
        mean += pair.first;
        pair.second = 1.0/num_examples;
      }

      // loop to compute std dev
      mean /= num_examples;
      for (auto &pair: training_data) {
        FeatureVector v = pair.first - mean;
        std = std.array() + (v.array() * v.array());
      }

      //std::cout << "Mean = " << mean.transpose() << "\n";

      // finish computing std dev
      //if (isStatic()) {
      std += EigenVectornd::Constant(std.size(),1,0.1);
      //}
      std /= training_data.size();
      std = std.cwiseSqrt();
      std = std.cwiseInverse();

      // 0 mean 1 variance
      for (auto &pair: training_data) {
        FeatureVector v = pair.first - mean;
        std::pair<FeatureVector,double> norm_pair(v.array()*std.array(),pair.second);
        normalized_training_data.push_back(norm_pair); // this could be more efficient
      }

      model = GmmPtr(new Gmm(dim,k));

      //model->Init(normalized_training_data.begin()->first,normalized_training_data.rbegin()->first);
      model->Init(EigenVectornd::Constant(dim,-1),EigenVectornd::Constant(dim,1));
      Matrixnd S = 1e-5*Matrixnd::Identity(dim,dim);
      model->Fit(normalized_training_data,1,50,&S);
      model->Update();

      // save the original matrices
      P.resize(k);
      for (unsigned int i = 0; i < k; ++i) {
        //model->ns[i].P += 1e-10*Matrixnd::Identity(dim,dim);
        P[i] = model->ns[i].P;
      }
      model->Update();

      //std::cout << "MODEL INFO >>>>>>>>" << std::endl;
      //std::cout << *model << std::endl;
      //std::cout << "Sigma^-1 = \n" << model->ns[0].Pinv << std::endl;
      //std::cout << "norm = " <<model->ns[0].norm << std::endl;
      //std::cout << "MODEL INFO >>>>>>>>" << std::endl;

    } else {
      std::cerr << __FILE__ << ":" << __LINE__ << ": Tried to train empty Skill!" << std::endl;
    }
  }

  /**
   * normalize a bunch of data before sending it through the GMM
   */
  void Skill::normalizeData(std::vector<FeatureVector> &data) {

    // apply normalization to each entry
    for (FeatureVector &vec: data) {
      //vec = (vec - mean).array() * std.array();
      vec -= mean;
      vec.array() *= std.array();
    }
  }

  /**
   * probabilities
   * what is the probability associated with each feature vector in the data?
   */
  FeatureVector Skill::logL(std::vector<FeatureVector> &data) {
    FeatureVector vec(data.size());

    if (model->k == 1) {
      for (unsigned int i = 0; i < data.size(); ++i) {
        vec(i) = log_prior + model->ns[0].logL(data[i]);
      }
    } else {
      for (unsigned int i = 0; i < data.size(); ++i) {
        vec(i) = log_prior + model->logL(data[i]);
      }
    }

    return vec;
  }

  /**
   * set up the name
   */
  Skill &Skill::setName(const std::string &name_) {
    name = name_;
    return *this;
  }

  /**
   * return name of this skill
   */
  const std::string &Skill::getName() const {
    return name;
  }

  /**
   * print gmm
   */
  void Skill::printGmm() {
    std::cout << *model << std::endl;
  }

  /**
   * add a particular object that should be attached
   * to the end of the kinematic chain for performing this motion
   */
  Skill &Skill::attachObject(const std::string &object) {
    attached_object = object;
    return *this;
  }

  /**
   * add model normalization to the different things
   */
  void Skill::addModelNormalization(const double &value) {
    unsigned int i = 0;
    unsigned int dim = model->ns[0].mu.size();
    for (auto &n: model->ns) {
      n.P += (value * Matrixnd::Identity(dim,dim));
      ++i;
    }
    model->Update();
  }

  /**
   * reset the model back to its original matrices
   */
  void Skill::resetModel() {
    unsigned int i = 0;
    for (auto &n: model->ns) {
      n.P = P[i];
      ++i;
    }
  }

  /**
   * is an object attached?
   */
  const bool Skill::hasAttachedObject() const {
    return attached_object.size() > 0;
  }

  /** 
   * what object is attached?
   */
  const std::string &Skill::getAttachedObject() const {
    return attached_object;
  }

}
