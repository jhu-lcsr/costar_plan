#ifndef _COLLISION_MAP
#define _COLLISION_MAP

#include <vector>
#include <unordered_map>

namespace costar {

  typedef std::vector<double> Position_t; // store a set of joint positions

  /**
   * CollisionMapHash
   * Hashes a collision map according to some parameters we set.
   */
  class CollisionMapHash {
  public:
      std::size_t operator()(Position_t const& pos) const;
  };


  /**
   * CollisionMap
   * Enable fast trajectory search by hashing collision detection calls within some scaling factor.
   *
   */
  class CollisionMap {
    private:
      std::unordered_map<Position_t,bool,CollisionMapHash> map; // store hashed collisions
      bool verbose;
      double factor;

    public:

      CollisionMap();

      void setVerbose(bool set);

      /**
       * reset()
       * This function clears the current map.
       */
      void reset();

      int check(Position_t const& pos) const;

      void update(Position_t const& pos, bool collision);
  };
}


#endif
