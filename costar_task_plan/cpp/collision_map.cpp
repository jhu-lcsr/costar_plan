#include <grid/collision_map.h>
#include <iostream>

namespace grid {

  std::size_t CollisionMapHash::operator()(Position_t const& pos) const {
    std::size_t hash = 0;
    for (unsigned int i = 0; i < pos.size(); ++i) {
      hash ^= (int)(20*pos[i]) << 2*i;
    }

    return hash;
  }



  void CollisionMap::reset() {
    map.clear();
  }

  int CollisionMap::check(Position_t const& pos) const {
    if (map.find(pos) != map.end()) {
      if (verbose) {
        std::cout << "Found: ";
        for (const double &q: pos) {
          std::cout << q << " ";
        }
        std::cout << std::endl;
      }
      return (int)map.at(pos);
    } else {
      return -1;
    }
  }

  void CollisionMap::update(Position_t const& pos, bool collision) {
    map[pos] = collision;
  }


  CollisionMap::CollisionMap() : verbose(true), factor(20.)
  {
  }

  void CollisionMap::setVerbose(bool set) {
    verbose = set;
  }
}
