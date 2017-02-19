#include <grid/training_features.h>

// read in poses from topics
#include <geometry_msgs/Pose.h>
#include <kdl_conversions/kdl_msg.h>
#include <iostream>
#include <sstream>

namespace grid {


  const std::vector<WorldConfiguration> &TrainingFeatures::getData() const {
    return data;
  }

  /**
   * get last pose
   */
  Pose TrainingFeatures::getPoseFrom(const std::string &name, FeatureVector f) {
    Pose fpose;
    int idx = 0;
    for (auto &pair: feature_types) {
      if (name == pair.first and pair.second == POSE_FEATURE) {
        featuresToPose(f,fpose,idx);
        break;
      } else {
        idx += feature_sizes[pair.first];
      }
    }
    return fpose;
  }

  /**
   * initialize training features with the necessary world objects to find
   */
  TrainingFeatures::TrainingFeatures(const std::vector<std::string> &objects_) :
    Features(), objects(objects_), min_t(0), max_t(0)
  {
    topics.push_back(JOINT_STATES_TOPIC);
    topics.push_back(BASE_TFORM_TOPIC);
    topics.push_back(GRIPPER_MSG_TOPIC);

    for (std::string &obj: objects) {
      std::stringstream ss;
      ss << "world/" << obj;
      topics.push_back(ss.str().c_str());
      topic_to_object[ss.str().c_str()]=obj;
    }
  }

  /**
   * print basic info for debugging
   */
  void TrainingFeatures::printTrainingFeaturesInfo() {
    std::cout << "Topics: " << std::endl;
    for (std::string &topic: topics) {
      std::cout << " - " << topic << std::endl;
    }

    std::cout << "Loaded " << data.size() << " data points." << std::endl;

    for (WorldConfiguration &conf: data) {
      std::cout << "Data point at t=" << conf.t << std::endl;
      std::cout << "\tBase at x=" << conf.base_tform.p.x()
        <<", y=" << conf.base_tform.p.y()
        <<", z=" << conf.base_tform.p.z()
        <<std::endl;
      std::cout <<"\tJoints = ";
      for (double &q: conf.joint_states.position) {
        std::cout << q << ", ";
      }
      std::cout << std::endl;
      std::cout <<"\tEnd at x=" << conf.ee_tform.p.x()
        <<", y=" << conf.ee_tform.p.y()
        <<", z=" << conf.ee_tform.p.z()
        <<std::endl;

      for(std::pair<const std::string,Pose> &obj: conf.object_poses) {
        std::cout <<"\tObject \"" << obj.first << "\" at x=" << obj.second.p.x()
          <<", y=" << obj.second.p.y()
          <<", z=" << obj.second.p.z()
          <<std::endl;

        Pose pose = obj.second.Inverse() * conf.base_tform * conf.ee_tform;
        std::cout << "\tComputed at x=" << pose.p.x()
          <<", y=" << pose.p.y()
          <<", z=" << pose.p.z()
          <<std::endl;

      }

    }
  }

  /* getPose
   * This function needs to be implemented by inheriting classes.
   * Time field helps determine when the query should occur.
   * A feature query gets the set of all featutes for different points in time, normalizes them, and returns.
   */
  std::vector<Pose> TrainingFeatures::getPose(const std::string &name,
                                              double mintime,
                                              double maxtime) {
    std::vector<Pose> poses;

    if(feature_types.find(name) == feature_types.end()
       or feature_types.at(name) != POSE_FEATURE)
    {
      ROS_ERROR("Feature with id \"%s\" is missing or not a pose!",name.c_str());
      ROS_ERROR("Exiting!");
      exit(-1);
    }

    for (WorldConfiguration &w: data) {
      if ((mintime == maxtime && maxtime == 0)
          or (w.t.toSec() > mintime && w.t.toSec() < maxtime))
      {
        // add this timestep to the data
        Pose pose = w.object_poses.at(name).Inverse() * w.base_tform * w.ee_tform;
        poses.push_back(pose);
      }
    }


    return poses;
  }


  /* getFeatureValues
   * Returns a list of features converted into a format we can use.
   */
  std::vector<FeatureVector> TrainingFeatures::getFeatureValues(const std::string &name,
                                                                double mintime,
                                                                double maxtime) {
    //std::vector<std::vector<double> > values;
    std::vector<FeatureVector> values(data.size());

    unsigned int i = 0;
    for (WorldConfiguration &w: data) {
      if ((mintime == maxtime && maxtime == 0)
          or (w.t.toSec() > mintime && w.t.toSec() < maxtime))
      {
        // add this timestep to the data
        if (i < 1) {
          FeatureVector f = worldToFeatures(w); // not quite right since this gets everything
          values[i] = f;
        } else {
          FeatureVector f = worldToFeatures(w,values[i-1]); // not quite right since this gets everything
          values[i] = f;
        }
        ++i;
      }
    }


    return values;
  }


  /**
   * getFeatureValues
   * get all available features from provided set
   */
  std::vector<FeatureVector> TrainingFeatures::getFeatureValues(const std::vector<std::string> &features) {
    std::vector<FeatureVector> values(data.size());

    unsigned int i = 0;
    for (WorldConfiguration &w: data) {
      if (i < 1) {
        FeatureVector f = worldToFeatures(w,features);
        values[i] = f;
      } else {
        FeatureVector f = worldToFeatures(w,features,values[i-1]); // not quite right since this gets everything
        values[i] = f;
      }
      ++i;
    }

    return values;
  }

  /**
   * get all available features
   * for testing, at least for now
   */
  std::vector<FeatureVector> TrainingFeatures::getAllFeatureValues() {
    std::vector<FeatureVector> values(data.size());

    unsigned int i = 0;
    for (WorldConfiguration &w: data) {
      if (i < 1) {
        FeatureVector f = worldToFeatures(w); // not quite right since this gets everything
        values[i] = f;
      } else {
        FeatureVector f = worldToFeatures(w,values[i-1]); // not quite right since this gets everything
        values[i] = f;
      }
      ++i;
    }

    return values;
  }


  /**
   * helper
   * convert a world into a set of features
   */
  FeatureVector TrainingFeatures::worldToFeatures(const WorldConfiguration &w, const std::vector<std::string> &features, FeatureVector &prev) const {
    FeatureVector f(getFeaturesSize(features,use_diff));
    unsigned int next_idx = 0;
    unsigned int pose_size = POSE_FEATURES_SIZE;
    if (not use_diff) {
      pose_size = POSE_FEATURES_SIZE_ND;
    }

    for (const std::string &feature: features) {
      if (feature_types.find(feature) != feature_types.end()) {
        if (feature_types.at(feature) == POSE_FEATURE) {

          Pose pose;
          if (not attached) {
            pose = w.object_poses.at(feature).Inverse() * w.base_tform * w.ee_tform;
          } else {
            //std::cout << "attached object: " << attachedObject << "\n";
            pose = w.object_poses.at(feature).Inverse() * w.object_poses.at(attachedObject); //w.ee_tform;
          }

          getPoseFeatures(pose,f,next_idx,prev,use_diff);
          next_idx += pose_size;

        } else if(feature_types.at(feature) == TIME_FEATURE) {
          f(next_idx) = (w.t - min_t).toSec() / (max_t - min_t).toSec();

        } else {
          next_idx += feature_sizes.at(feature);
        }
      } else {
        std::cerr << __FILE__ << ":" << __LINE__ << ": missing feature: " << feature << std::endl;
      }
    }

    return f;
  }

  /**
   * helper
   * convert a world into a set of features
   */
  FeatureVector TrainingFeatures::worldToFeatures(const WorldConfiguration &w,FeatureVector &prev) const {

    using KDL::Rotation;

    FeatureVector f(getFeaturesSize(),use_diff);
    unsigned int next_idx = 0;

    for (std::pair<const std::string, FeatureType> pair: feature_types) {
      if (pair.second == POSE_FEATURE) {

        Pose pose = w.object_poses.at(pair.first).Inverse() * w.base_tform * w.ee_tform;
        std::cout << pair.first << " " << next_idx << "/" << getFeaturesSize()<< "\n";
        getPoseFeatures(pose,f,next_idx,prev,use_diff);
        next_idx += POSE_FEATURES_SIZE;

      } else if(feature_types.at(pair.first) == TIME_FEATURE) {
        f(next_idx) = (w.t.toSec() - min_t.toSec()) / (max_t.toSec() - min_t.toSec());
        //std::cout << __LINE__ << ": t=" << f(next_idx) << "; " << max_t.toSec() << "; " << min_t.toSec() << std::endl;

      } else {
        next_idx += feature_sizes.at(pair.first);
      }
    }

    return f;
  }


  /**
   * helper
   * convert a world into a set of features
   */
  FeatureVector TrainingFeatures::worldToFeatures(const WorldConfiguration &w, const std::vector<std::string> &features) const {
    FeatureVector f(getFeaturesSize(features,use_diff));
    unsigned int next_idx = 0;
    unsigned int pose_size = POSE_FEATURES_SIZE;
    if (not use_diff) {
      pose_size = POSE_FEATURES_SIZE_ND;
    }

    for (const std::string &feature: features) {
      if (feature_types.find(feature) != feature_types.end()) {
        if (feature_types.at(feature) == POSE_FEATURE) {

          Pose pose;
          if (not attached) {
            pose = w.object_poses.at(feature).Inverse() * w.base_tform * w.ee_tform;
          } else {
            //std::cout << "attached object: " << attachedObject << "\n";
            pose = w.object_poses.at(feature).Inverse() * w.object_poses.at(attachedObject); //w.ee_tform;
          }

          getPoseFeatures(pose,f,next_idx,use_diff);
          next_idx += pose_size;

        } else if(feature_types.at(feature) == TIME_FEATURE) {
          f(next_idx) = (w.t - min_t).toSec() / (max_t - min_t).toSec();

        } else {
          next_idx += feature_sizes.at(feature);
        }
      } else {
        std::cerr << __FILE__ << ":" << __LINE__ << ": missing feature: " << feature << std::endl;
      }
    }

    return f;
  }

  /**
   * helper
   * convert a world into a set of features
   */
  FeatureVector TrainingFeatures::worldToFeatures(const WorldConfiguration &w) const {

    using KDL::Rotation;

    FeatureVector f(getFeaturesSize(),use_diff);
    unsigned int next_idx = 0;

    for (std::pair<const std::string, FeatureType> pair: feature_types) {
      if (pair.second == POSE_FEATURE) {

        Pose pose = w.object_poses.at(pair.first).Inverse() * w.base_tform * w.ee_tform;
        getPoseFeatures(pose,f,next_idx,use_diff);
        next_idx += POSE_FEATURES_SIZE;

      } else if(feature_types.at(pair.first) == TIME_FEATURE) {
        f(next_idx) = (w.t.toSec() - min_t.toSec()) / (max_t.toSec() - min_t.toSec());
        //std::cout << __LINE__ << ": t=" << f(next_idx) << "; " << max_t.toSec() << "; " << min_t.toSec() << std::endl;

      } else {
        next_idx += feature_sizes.at(pair.first);
      }
    }

    return f;
  }

  /**
   * read
   * Open a rosbag containing the demonstrations.
   * We make some assumptions as to how these are stored.
   * This function will read in the poses and other information associated with the robot.
   * This information all gets stored and can be used to compute features or retrieve world configurations.
   */
  void TrainingFeatures::read(const std::string &bagfile, int downsample) {
    using rosbag::View;
    using rosbag::MessageInstance;
    typedef geometry_msgs::Pose PoseMsg;

    bag.open(bagfile, rosbag::bagmode::Read);

    View view(bag, rosbag::TopicQuery(topics));

    WorldConfiguration conf;
    conf.t = ros::Time(0);

    int count = 0;
    // loop over all messages
    for(const MessageInstance &m: view) {

      //std::cout << "[READING IN] " << m.getTopic() << std::endl;

      if (conf.t == ros::Time(0)) {
        conf.t = m.getTime();
      } else if (conf.t != m.getTime()) {

        ++count;
        if (not (downsample and count % downsample != 0)) { 
          data.push_back(conf);
        }
        conf = WorldConfiguration();
        conf.t = m.getTime();
      }

      // track min and max times
      if (conf.t < min_t || min_t == ros::Time(0)) {
        min_t = conf.t;
      }
      if (conf.t > max_t || max_t == ros::Time(0)) {
        max_t = conf.t;
      }

      if (m.getTopic() == GRIPPER_MSG_TOPIC) {
        addFeature(GRIPPER_CMD,FLOAT_FEATURE);
        conf.gripper_cmd = getGripperFeatures(m);
      } else if (m.getTopic() == JOINT_STATES_TOPIC) {

        // save joints
        using sensor_msgs::JointState;
        conf.joint_states = *m.instantiate<JointState>();

        // compute forward kinematics
        conf.ee_tform = robot->FkPos(conf.joint_states.position);

      } else if (m.getTopic() == BASE_TFORM_TOPIC) {

        // convert from message to KDL pose
        PoseMsg msg = *m.instantiate<PoseMsg>();
        Pose p;
        tf::poseMsgToKDL(msg,p);
        conf.base_tform = p;

      } else if (topic_to_object.find(m.getTopic()) != topic_to_object.end()) {

        conf.gripper_cmd = getGripperFeatures(m);
        // convert from message to KDL pose
        PoseMsg msg = *m.instantiate<PoseMsg>();
        Pose p;
        tf::poseMsgToKDL(msg,p);

        // assign data
        addFeature(topic_to_object[m.getTopic()],POSE_FEATURE);
        conf.object_poses[topic_to_object[m.getTopic()]] = p;
      }
    }

    bag.close();

  }

  /**
   * return the joint states data we care about
   */
  Pose TrainingFeatures::getJointStatesData(rosbag::MessageInstance const &m) {
    Pose p;
    std::cerr << __FILE__ << ":" << __LINE__ << ": function not implemented!" << std::endl;

    return p;
  }

  /**
   * get the object poses we care about
   */
  Pose TrainingFeatures::getObjectData(rosbag::MessageInstance const &m) {
    Pose p;
    std::cerr << __FILE__ << ":" << __LINE__ << ": function not implemented!" << std::endl;

    return p;
  }

  /**
   * print all extractable features for the different objects
   */
  void TrainingFeatures::printExtractedFeatures() {
    std::cerr << __FILE__ << ":" << __LINE__ << ": function not implemented!" << std::endl;
  }

#if 0
  /*
   * return the gripper features from a rosbag
   * default: returns nothing
   */
  virtual std::vector<double> TrainingFeatures::getGripperFeatures(rosbag::MessageInstance const &m) {
    return std::vector<double>();
  }
#endif

}
