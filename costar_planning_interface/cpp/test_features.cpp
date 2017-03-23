#include <costar_task_plan/test_features.h>

//#define DEBUG_PRINT_TF_POSE

namespace costar {

  /* getPose
   * This function needs to be implemented by inheriting classes.
   * Time field helps determine when the query should occur.
   * A feature query gets the set of all featutes for different points in time, normalizes them, and returns.
   */
  TrajectoryFrames TestFeatures::getPose(const std::string &name,
                                         double mintime,
                                         double maxtime) {
    TrajectoryFrames poses;


    return poses;
  }

  /* getFeatureValues
   * Returns a list of features converted into a format we can use.
   */
  std::vector< FeatureVector > TestFeatures::getFeatureValues(const std::string &name,
                                                              double mintime,
                                                              double maxtime) {
    std::vector< FeatureVector > values;

    return values;
  }

  /* setFrame
   * Adds a frame of reference as a feature
   */
  TestFeatures &TestFeatures::setFrame(const std::string &frame, const std::string &objectClass) {
    objectClassToID[objectClass] = frame;
    return *this;
  }

  /* addAgent:
   * configure agent's manipulation frame
   */
  TestFeatures &TestFeatures::setAgentFrame(const std::string &agentFrame_) {
    agentFrame = agentFrame_;
    objectClassToID[AGENT] = agentFrame_;
    addFeature(AGENT,POSE_FEATURE);
    return *this;
  }

  /* addBase:
   * configure base's manipulation frame
   */
  TestFeatures &TestFeatures::setBaseFrame(const std::string &baseFrame_) {
    baseFrame = baseFrame_;
    objectClassToID[BASE] = baseFrame_;
    addFeature(BASE,POSE_FEATURE);
    return *this;
  }

  /* configure world frame for this TestFeatures object
  */
  TestFeatures &TestFeatures::setWorldFrame(const std::string &worldFrame_) {
    worldFrame = worldFrame_;
    return *this;
  }

  /* lookup tf frame for key
   * in world frame
   */
  Pose TestFeatures::lookup(const std::string &key) {
    return lookup(key, worldFrame);
  }

  /* lookup tf frame for key
  */
  Pose TestFeatures::lookup(const std::string &key, const std::string &in_frame) {
    tf::StampedTransform transform;
    Pose p;

    try{
#ifdef DEBUG_PRINT_TF_POSE
      std::cout << "looking up " << objectClassToID[key] << " for " << key << std::endl;
#endif
      listener.lookupTransform(in_frame, objectClassToID[key],
                               ros::Time(0), transform);
      tf::transformTFToKDL(transform, p);
#ifdef DEBUG_PRINT_TF_POSE
      std::cout << "[" << key << "] x = " << transform.getOrigin().getX() << std::endl;
      std::cout << "[" << key << "] y = " << transform.getOrigin().getY() << std::endl;
      std::cout << "[" << key << "] z = " << transform.getOrigin().getZ() << std::endl;
#endif
    }
    catch (tf::TransformException ex){
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      std::cerr << ex.what() << std::endl;
      std::cerr << "With key = " << key << std::endl;
      std::cerr << "With world = " << worldFrame << std::endl;
      ROS_ERROR("%s with key \"%s\"",ex.what(),key.c_str());
    }

    return p;
  }

  /**
   * get the current end effector position
   */
  Pose TestFeatures::getCurrentEndEffector() const {
    std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
    return currentPose.at(BASE) * currentPose.at(AGENT);
  }

  /**
   * get the world frame
   */
  const std::string &TestFeatures::getWorldFrame() const {
    return worldFrame;
  }

  /*
   * run lookup for all objects
   * store results for poses from tf
   */
  TestFeatures &TestFeatures::updateWorldfromTF() {
    for (const std::pair<std::string,FeatureType> &feature: feature_types) {
      //std::cout << feature.first << ", " << feature.second << std::endl;
      //if (feature.first == AGENT) {
      //  currentPose[feature.first] = lookup(feature.first,baseFrame);
      //} else
      if(feature.second == POSE_FEATURE) {
        //std::cout << feature.first << std::endl;
        currentPose[feature.first] = lookup(feature.first);
        currentPoseInv[feature.first] = currentPose[feature.first].Inverse();
      }

    }

    if (attached) {
      attachedObjectFrame = lookup(AGENT).Inverse() * lookup(attachedObject);
      //attachedObjectFrame = lookup(AGENT,attachedObject);
      //std::cout << attachedObjectFrame << "\n";
    } else {
      attachedObjectFrame = Pose();
    }

    return *this;
  }

  /* getFeaturesForTrajectory
   * Get information for a single feature over the whole trajectory given in traj.
   * Traj is KDL::Trajectory
   */
  void TestFeatures::getFeaturesForTrajectory(std::vector<FeatureVector> &features,
                                              const std::vector<std::string> &names,
                                              Trajectory *traj, double dt)
  {

    using KDL::Rotation;

    features.resize((unsigned int)1+floor(traj->Duration() / dt));
    unsigned int next_idx = 0;
    unsigned int dim = getFeaturesSize(names,use_diff);
    unsigned int pose_size = POSE_FEATURES_SIZE;
    if (not use_diff) {
      pose_size = POSE_FEATURES_SIZE_ND;
    }

    for (double t = 0; t < traj->Duration(); t += dt) {
      unsigned int idx = 0;
      FeatureVector f(dim);
      for (const std::string &name: names) {

        if (feature_types[name] == POSE_FEATURE) {
          Pose offset = currentPoseInv[name] * traj->Pos(t);
          if (attached) {
            offset = offset * attachedObjectFrame;
          }

          //std::cout << "\tComputed at x=" << offset.p.x()
          //  <<", y=" << offset.p.y()
          //  <<", z=" << offset.p.z()
          //  <<std::endl;

          //std::cout << __LINE__ << ": " << dim << ", " << idt << std::endl;
          if (next_idx == 0) {
            getPoseFeatures(offset,f,idx,use_diff);
          } else {
            getPoseFeatures(offset,f,idx,features[next_idx-1],use_diff);
          }

          idx+= pose_size;//POSE_FEATURES_SIZE;

        } else if (feature_types[name] == TIME_FEATURE) {
          f(idx) = t / traj->Duration();
          idx += TIME_FEATURES_SIZE;
        }
      }

#if 0
      for (int i = 0; i < dim; ++i) {
        std::cout << f(i) << " ";
      } 
      std::cout << std::endl;
#endif

      features[next_idx++] = f;
      assert(idx == dim);
    }
    assert(next_idx == features.size());
  }

  /* getFeaturesForTrajectory
   * Get information for a single feature over the whole trajectory given in traj.
   * Traj is a set of frames
   * Uses an attached object frame
   */
  void TestFeatures::getFeaturesForTrajectory(std::vector<FeatureVector> &features,
                                              const std::vector<std::string> &names,
                                              const TrajectoryFrames &traj,
                                              const bool useAttachedObjectFrame,
                                              const Pose &attachedObjectFrame)
  {

    features.resize(traj.size());

    unsigned int next_idx = 0;
    unsigned int dim = getFeaturesSize(names,use_diff);
    unsigned int pose_size = POSE_FEATURES_SIZE;
    if (not use_diff) {
      pose_size = POSE_FEATURES_SIZE_ND;
    }

    for (const Pose &p: traj) {
      unsigned int idx = 0;
      features[next_idx].resize(dim);
      //FeatureVector f(dim);

      for (const std::string &name: names) {

        if (feature_types[name] == POSE_FEATURE) {

          Pose offset = currentPoseInv[name] * p;
          if (useAttachedObjectFrame) {
            offset = offset * attachedObjectFrame;
          }

          if (next_idx==0) {
            getPoseFeatures(offset,features[next_idx],idx,use_diff);
          } else {
            getPoseFeatures(offset,features[next_idx],idx,features[next_idx-1],use_diff);
          }
          idx+= pose_size;

        } else if (feature_types[name] == TIME_FEATURE) {
          features[next_idx](idx) = (double)next_idx / (double)traj.size();
          idx += TIME_FEATURES_SIZE;
        }

      }
      //features[next_idx++] = f;
      next_idx++;

    }
  }



  /* getFeaturesForTrajectory
   * Get information for a single feature over the whole trajectory given in traj.
   * Traj is a set of frames
   */
  void TestFeatures::getFeaturesForTrajectory(std::vector<FeatureVector> &features,
                                              const std::vector<std::string> &names,
                                              const TrajectoryFrames &traj)
  {
    getFeaturesForTrajectory(features, names, traj, attached, attachedObjectFrame);
  }


  /**
   * get current attached object frame
   */
  const Pose &TestFeatures::getAttachedObjectFrame() const {
    return attachedObjectFrame;
  }


  std::vector<std::string> TestFeatures::getClasses() const {
    std::vector<std::string> out;
    for (const std::pair<std::string,FeatureType> &elem: feature_types) {
      if (elem.second == POSE_FEATURE) {
        out.push_back(elem.first);
      }
    }
    return out;
  }

  /**
   * get all the current IDs (coordinate frames)
   */
  std::vector<std::string> TestFeatures::getIds() const {
    std::vector<std::string> out;
    for (const std::pair<std::string,FeatureType> &elem: feature_types) {
      if (elem.second == POSE_FEATURE) {
        out.push_back(objectClassToID.at(elem.first));
      }
    }
    return out;
  }
}
