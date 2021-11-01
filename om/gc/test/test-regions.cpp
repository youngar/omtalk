#include <ab/Util/Assert.h>
#include <ab/Util/BitArray.h>
#include <catch2/catch.hpp>
#include <cstddef>
#include <om/GC/ForwardingMap.h>
#include <om/GC/Heap.h>
#include <om/GC/Ref.h>
#include <type_traits>

using namespace om::gc;

namespace om::gc {

struct RegionMapChecks {
  static_assert(std::is_trivially_destructible_v<RegionMap>);
  // static_assert(sizeof(RegionMap) == (REGION_MAP_NCHUNKS *
  // sizeof(ab::BitChunk)));
  static_assert(
      ab::check_size<RegionMap, (REGION_MAP_NCHUNKS * sizeof(ab::BitChunk))>());
};

static_assert(sizeof(FreeBlock) == MIN_OBJECT_SIZE);

class RegionChecks {
  static_assert((sizeof(Region) % OBJECT_ALIGNMENT) == 0);
  static_assert((offsetof(Region, data) % OBJECT_ALIGNMENT) == 0);
  static_assert(ab::check_size<Region, REGION_SIZE>());
  static_assert(sizeof(Region) <= REGION_SIZE);
};

} // namespace om::gc

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

TEST_CASE("Count markmap bits", "[MarkMap]") {
  RegionManager regionManager;
  Region *region = regionManager.allocateRegion();

  region->clearMarkMap();
  // REQUIRE(region->)

  REQUIRE(region->mark(region->heapBegin() + 0x100) == true);
  REQUIRE(region->mark(region->heapBegin() + 0x300) == true);
  // REQUIRE()
}
