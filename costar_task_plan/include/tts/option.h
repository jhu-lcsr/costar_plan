
#include "tts/types.h"
#include "tts/goal.h"
#include "tts/action.h"

namespace task_tree_search {

// Describe a single state, including both a goal and a path to get there.
class Option {
  Goal::Ptr goal;
  Action::Ptr action;
};


} // task_tree_search
