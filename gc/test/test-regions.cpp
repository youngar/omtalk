#include <catch2/catch.hpp>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>
#include <omtalk/Util/Assert.h>
#include <omtalk/RegionManager.h>

using namespace omtalk::gc;
namespace ot = omtalk;
namespace gc = ot::gc;

namespace omtalk::gc {

struct RegionMapChecks {
  static_assert(std::is_trivially_destructible_v<RegionMap>);
  // static_assert(sizeof(RegionMap) == (REGION_MAP_NCHUNKS *
  // sizeof(BitChunk)));
  static_assert(
      check_size<RegionMap, (REGION_MAP_NCHUNKS * sizeof(BitChunk))>());
};

static_assert(sizeof(FreeBlock) == MIN_OBJECT_SIZE);

class RegionChecks {
  static_assert((sizeof(Region) % OBJECT_ALIGNMENT) == 0);
  static_assert((offsetof(Region, data) % OBJECT_ALIGNMENT) == 0);
  static_assert(check_size<Region, REGION_SIZE>());
  static_assert(sizeof(Region) <= REGION_SIZE);
};

} // namespace omtalk::gc

TEST_CASE("ctor", "[RegionManager]") {
  gc::RegionManager manager;
}

TEST_CASE("Mark Map works", "[MarkMap]") {
  RegionManager regionManager;
  Region *region = regionManager.allocate();
  REQUIRE(regionManager.managed(region));
  REQUIRE(regionManager.managed(region + REGION_SIZE - 1));
  regionManager.free(region);
}
