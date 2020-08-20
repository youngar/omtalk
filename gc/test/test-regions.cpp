#include <catch2/catch.hpp>
#include <omtalk/Heap.h>
#include <omtalk/Ref.h>

using namespace omtalk::gc;

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
