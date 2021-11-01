#include <ab/Util/ValueIter.h>
#include <array>
#include <catch2/catch.hpp>
#include <cstdint>
#include <om/GC/ForwardingMap.h>

using namespace om;
using namespace om::gc;

TEST_CASE("", "[ForwardingEntry]") {
  ForwardingEntry e;

  SECTION("initial value") {
    REQUIRE(e.get() == nullptr);
    REQUIRE(e.isLocked() == false);
  }

  SECTION("try locking") {
    REQUIRE(e.tryLock() == true);
    REQUIRE(e.isLocked() == true);
    REQUIRE(e.tryLock() == false);
  }

  SECTION("set and verify") {
    REQUIRE(e.tryLock() == true);
    int v;
    e.set(&v);
    REQUIRE(e.get() == &v);
  }
}

TEST_CASE("zero sized", "[FowardingMap]") {
  ForwardingMap map;
  REQUIRE(map.size() == 0);
  map.clear();
  REQUIRE(map.size() == 0);
}

TEST_CASE("multiple elements", "[FowardingMap]") {
  ForwardingMap map;
  std::array<std::uintptr_t, 10> heap;
  std::vector<void *> from{&heap[0], &heap[2], &heap[3]};
  std::vector<void *> to{&heap[5], &heap[6], &heap[10]};

  map.rebuild(from.begin(), from.end());
  REQUIRE(map.size() == 3);

  for (std::size_t i = 0; i < from.size(); i++) {
    auto &entry = map[from[i]];
    entry.tryLock();
    entry.set(to[i]);
  }

  for (std::size_t i = 0; i < from.size(); i++) {
    auto &entry = map[from[i]];
    REQUIRE(entry.get() == to[i]);
  }
}
