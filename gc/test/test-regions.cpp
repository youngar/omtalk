#include <catch2/catch.hpp>
#include <cstddef>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>
#include <omtalk/Util/Assert.h>
#include <omtalk/Util/BitArray.h>
#include <type_traits>

using namespace omtalk::gc;

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

TEST_CASE("Mark Map works", "[MarkMap]") {
  RegionManager regionManager;
  Region *region = regionManager.allocateRegion();

  region->clearMarkMap();

  Ref<void> address = region->heapBegin() + 0x100;

  REQUIRE(region->marked(address) == false);

  REQUIRE(region->mark(address) == true);

  REQUIRE(region->marked(address) == true);

  REQUIRE(region->unmark(address) == true);

  REQUIRE(region->marked(address) == false);
}
