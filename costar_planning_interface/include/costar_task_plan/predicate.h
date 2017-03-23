#ifndef _GRID_PREDICATE
#define _GRID_PREDICATE

#include <string>
#include <vector>

namespace costar {

  /**
   * Defines a PREDICATE.
   * Predicates are tokens that can appear in our grammars.
   * They control task flow, and are a part of the state of the world.
   *
   * If a predicate has a TEST associated with it, we can use this to detect failures.
   */
  class Predicate {
    protected:
      std::string name;

    public:

      Predicate(const std::string &name, const std::vector<std::string> &args);

  };

}

#endif
