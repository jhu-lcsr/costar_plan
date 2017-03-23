
#include "tts/trajectory.h"

namespace costar_task_plan {

  // Describe a single state, including both a goal and a path to get there.
  class Goal {

    // Estimate the value of a particular trajectory policy.
    virtual double score(const Trajectory& traj) = 0;

    // Is this an acceptable trajectory to follow?
    virtual bool accept(const InstantiatedTrajectory& traj) = 0;

    // Check a set of params to see if they result in an acceptable trajectory.
    virtual bool accept(const Trajectory& traj) {
      return accept(*traj.instantiate());
    }
  };

} // costar_task_plan
