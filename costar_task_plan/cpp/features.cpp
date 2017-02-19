#include<grid/features.h>

namespace grid {

  /**
   * Do we need to use diff features to train this?
   */
  void Features::setUseDiff(const bool _use_diff) {
    use_diff = _use_diff;
  }

  /** 
   * initializers
   */
  Features::Features() : attached(false), attachedObjectFrame(), attachedObject(""), use_diff(true)
  {

  }

  std::vector< FeatureVector > Features::getFeatures(std::vector<std::string> &names) {

    std::vector< std::vector <double> > f;
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    for (const std::string &name: names) {

    }
  }

  /**
   * add a feature
   * also updates expected array size
   */
  Features &Features::addFeature(const std::string &name, const FeatureType type, unsigned int size) {
    feature_types[name] = type;
    if (type == POSE_FEATURE) {
      feature_sizes[name] = POSE_FEATURES_SIZE;
      feature_sizes_nd[name] = POSE_FEATURES_SIZE_ND;
    } else if (type == TIME_FEATURE) {
      feature_sizes[name] = TIME_FEATURES_SIZE;
      feature_sizes_nd[name] = TIME_FEATURES_SIZE;
    } else {
      feature_sizes[name] = size;
      feature_sizes_nd[name] = size;
    }
    updateFeaturesSize();

    return *this;
  }


  /**
   * is there an object here
   */
  bool Features::hasAttachedObjectFrame() const {
    return attached;
  }


  /**
   * set all values in the vector to something
   */
  void Features::setAll(std::vector<FeatureVector> &features,
                        const std::vector<std::string> &all_features,
                        const std::string &name,
                        const double &value)
  {
    unsigned int size = feature_sizes[name];
    unsigned int idx = 0;

    for (const std::string &fname: all_features) {
      if (fname == name) {
        for (FeatureVector &vec: features) {
          for (unsigned int i = 0; i < size; ++i) {
            //std::cout << vec(idx+i) << " at " << (idx+i);
            vec(idx+i) = value;
            //std::cout << " = " << vec(idx+i) << std::endl;
          }
        }
        break;
      } else {
        idx += feature_sizes[fname];
      }
    }
  }

  /**
   * get the type of a feature
   */
  FeatureType Features::getFeatureType(const std::string &name) const {
    return feature_types.at(name);
  }

  /**
   * updateFeaturesSize
   * Compute number of features we expect to see
   */
  void Features::updateFeaturesSize() {
    features_size = 0;
    features_size_nd = 0;
    for (std::pair<const std::string,unsigned int> &pair: feature_sizes) {
      features_size += pair.second;
    }
    for (std::pair<const std::string,unsigned int> &pair: feature_sizes_nd) {
      features_size_nd += pair.second;
    }
  }

  /**
   * getFeaturesSize
   * Compute number of features we expect to see
   */
  unsigned int Features::getFeaturesSize() const {
    return features_size;
  }

  /**
   * getFeaturesSize
   * Compute number of features we expect to see
   */
  unsigned int Features::getFeaturesSize(const std::string &name) const {
    if (feature_sizes.find(name) != feature_sizes.end()) {
      return feature_sizes.at(name);
    } else {
      std::cerr << __FILE__ << ":" << __LINE__ << ": Unrecognized feature: " << name << std::endl;
      return 0;
    }
  }

  /**
   * getFeaturesSize
   * Compute number of features we expect to see
   */
  unsigned int Features::getFeaturesSize(const std::vector<std::string> &names) const {
    unsigned int size = 0;
    for (const std::string &name: names) {
      if (feature_sizes.find(name) != feature_sizes.end()) {
        size += feature_sizes.at(name);
      } else {
        std::cerr << __FILE__ << ":" << __LINE__ << ": Unrecognized feature: " << name << std::endl;
      }
    }
    return size;
  }

  /**
   * getFeaturesSize
   * Compute number of features we expect to see
   */
  unsigned int Features::getFeaturesSize(const std::vector<std::string> &names,
                                         bool use_diff) const
  {
    if (use_diff) {
      return getFeaturesSize(names);
    } else {
      return getFeaturesSizeNoDiff(names);
    }
  }


  /**
   * getFeaturesSizeNoDiff
   */
  unsigned int Features::getFeaturesSizeNoDiff(const std::vector<std::string> &names) const {
    unsigned int size = 0;
    for (const std::string &name: names) {
      if (feature_sizes_nd.find(name) != feature_sizes_nd.end()) {
        size += feature_sizes_nd.at(name);
      } else {
        std::cerr << __FILE__ << ":" << __LINE__ << ": Unrecognized feature: " << name << std::endl;
      }
    }
    return size;
  }

  /** 
   * setRobotKinematics
   * sets the thing that will actually compute forward and inverse kinematics for our robot
   */
  void Features::setRobotKinematics(std::shared_ptr<RobotKinematics> rk) {
    robot = rk;
    n_dof = robot->getDegreesOfFreedom();
  }

  /**
   * getPoseFeatures
   * Load pose data at index
   */
  void Features::getPoseFeatures(const Pose &pose, FeatureVector &f, unsigned int idx, bool use_diff) {

    f(idx+POSE_FEATURE_X) = pose.p.x();
    f(idx+POSE_FEATURE_Y) = pose.p.y();
    f(idx+POSE_FEATURE_Z) = pose.p.z();
    if (use_diff) {
      f(idx+POSE_FEATURE_DX) = 0;
      f(idx+POSE_FEATURE_DY) = 0;
      f(idx+POSE_FEATURE_DZ) = 0;
      f(idx+POSE_FEATURE_DDIST) = 0;
    }
#ifdef USE_ROTATION_RPY
    pose.M.GetRPY(f(idx+POSE_FEATURE_ROLL), f(idx+POSE_FEATURE_PITCH), f(idx+POSE_FEATURE_YAW));
#else
    pose.M.GetQuaternion(f(idx+POSE_FEATURE_WX),f(idx+POSE_FEATURE_WY),f(idx+POSE_FEATURE_WZ),f(idx+POSE_FEATURE_WW));
#endif
    f(idx+POSE_FEATURE_DIST)  = pose.p.Norm();
  }

  void Features::getPoseFeatures(const Pose &pose, FeatureVector &f, unsigned int idx, FeatureVector &prev, bool use_diff) {

    f(idx+POSE_FEATURE_X) = pose.p.x();
    f(idx+POSE_FEATURE_Y) = pose.p.y();
    f(idx+POSE_FEATURE_Z) = pose.p.z();
    if (use_diff) {
      f(idx+POSE_FEATURE_DX) = pose.p.x() - prev(idx+POSE_FEATURE_X);
      f(idx+POSE_FEATURE_DY) = pose.p.y() - prev(idx+POSE_FEATURE_Y);
      f(idx+POSE_FEATURE_DZ) = pose.p.z() - prev(idx+POSE_FEATURE_Z);
      f(idx+POSE_FEATURE_DDIST) = pose.p.Norm() - prev(idx+POSE_FEATURE_DIST);
    }
#ifdef USE_ROTATION_RPY
    pose.M.GetRPY(f(idx+POSE_FEATURE_ROLL), f(idx+POSE_FEATURE_PITCH), f(idx+POSE_FEATURE_YAW));
#else
    pose.M.GetQuaternion(f(idx+POSE_FEATURE_WX),f(idx+POSE_FEATURE_WY),f(idx+POSE_FEATURE_WZ),f(idx+POSE_FEATURE_WW));
#endif
    f(idx+POSE_FEATURE_DIST)  = pose.p.Norm();
  }

  /**
   * featuresToPose
   */
  void Features::featuresToPose(FeatureVector &f, Pose &p, unsigned int idx) {

    using KDL::Rotation;
    using KDL::Vector;

#ifdef USE_ROTATION_RPY
    Rotation rot = Rotation::RPY(f[POSE_FEATURE_ROLL], f[POSE_FEATURE_PITCH], f[POSE_FEATURE_YAW]);
#else
    Rotation rot = Rotation::Quaternion(f[POSE_FEATURE_WX],f[POSE_FEATURE_WY],f[POSE_FEATURE_WZ],f[POSE_FEATURE_WW]);
    //Rotation rot = Rotation::Quaternion(0,0,0,1);
#endif
    Vector vec = Vector(f[POSE_FEATURE_X],f[POSE_FEATURE_Y],f[POSE_FEATURE_Z]);
    p = Pose(rot,vec);
  }

  /**
   * attach an object frame to this set of features by identifier
   */
  Features &Features::attachObjectFrame(const std::string &object) {
    attached = true;
    attachedObject = object;
    return *this;
  }

  /**
   * detach all object frames
   */
  Features &Features::detachObjectFrame() {
    attachedObject = std::string("");
    return *this;
  }

  /**
   * return the name of an attached object
   */
  const std::string &Features::getAttachedObject() const {
    return attachedObject;
  }
}
