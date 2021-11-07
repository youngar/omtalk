#include <ab/Util/Atomic.h>
#include <catch2/catch.hpp>
#include <om/Om/MemoryManager.h>
#include <om/Om/ObjectProxy.h>

TEST_CASE("Object Model", "[adfadsf]") {
  auto mm = om::gc::MemoryManagerBuilder<om::om::Scheme>()
                .withRootWalker(
                    std::make_unique<om::gc::RootWalker<om::om::Scheme>>())
                .build();
}
