#include <grid/wam_training_features.h>
#include <oro_barrett_msgs/BHandCmd.h>

using oro_barrett_msgs::BHandCmd;

namespace grid {

  /**
   * initialize training features with the necessary world objects to find
   */
  WamTrainingFeatures::WamTrainingFeatures(
      const std::vector<std::string> &objects)
    : TrainingFeatures(objects) 
  {
  }

  /*
   * return the gripper features from a rosbag
   */
  std::vector<double> WamTrainingFeatures::getGripperFeatures(rosbag::MessageInstance const &m) {
    std::vector<double> features;

    BHandCmd::ConstPtr cmd = m.instantiate<BHandCmd>();

    if (cmd != NULL) {
      // extract the appropriate command variables
      features.push_back(cmd->cmd[0]);
      features.push_back(cmd->cmd[1]);
      features.push_back(cmd->cmd[2]);
    }

    return features;
  }

}
