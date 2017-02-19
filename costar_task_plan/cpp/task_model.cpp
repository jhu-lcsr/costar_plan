#include <grid/task_model.h>

namespace grid {

  /**
   * create an empty task model
   */
  TaskModel::TaskModel() {

  }

  /**
   * add an object with a particular class
   */
  TaskModel &TaskModel::addObject(const std::string &obj, const std::string &obj_class) {

    std::cout << __FILE__ << ":" << __LINE__ << ": Not yet implemented!" << std::endl;

    return *this;
  }

  /**
   * Enable transitions between different skills
   * This add is only in one direction from skill1 to skill2
   */
  TaskModel &TaskModel::setTransition(const Skill &skill1, const Skill &skill2, bool allowed) {

    if (transitions.size() < skills.size()) {
      transitions.resize(skills.size());
    }
    for (auto &vec: transitions) {
      if (vec.size() < skills.size()) {
        vec.resize(skills.size());
      }
    }

    unsigned int idx1 = skill_to_id[skill1.getName()];
    unsigned int idx2 = skill_to_id[skill2.getName()];

    transitions[idx1][idx2] = allowed;

    return *this;
  }

  /**
   * add an uninstantiated skill
   * appends it to skills vector
   * adds a reference in skills to id mapping
   */
  TaskModel &TaskModel::addSkill(const Skill &skill) {
    if (skill_to_id.find(skill.getName()) != skill_to_id.end()) {
      skill_to_id[skill.getName()] = skills.size();
      skills.push_back(skill);
    }
    return *this;
  }

}
