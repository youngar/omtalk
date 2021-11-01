#include <ab/Util/Bytes.h>
#include <catch2/catch.hpp>
#include <cstring>
#include <example/Object.h>
#include <memory>
#include <om/GC/GlobalCollector.h>
#include <om/GC/Heap.h>
#include <om/GC/MemoryManager.h>
#include <om/GC/Ref.h>

using namespace om;
using namespace om::gc;

TEST_CASE("Startup and Shutdown MemoryManager", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
}

TEST_CASE("Startup and Shutdown MemoryManager and Contexts",
          "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  REQUIRE(mm.getContextCount() == 0);
  Context<TestCollectorScheme> context1(mm);
  REQUIRE(mm.getContextCount() == 1);
  Context<TestCollectorScheme> context2(mm);
  REQUIRE(mm.getContextCount() == 2);
  Context<TestCollectorScheme> context3(mm);
  REQUIRE(mm.getContextCount() == 3);
}

TEST_CASE("Initial memory allocation", "[garbage collector]") {
  MemoryManagerConfig config;

  SECTION("0 initial memory") {
    config.initialMemory = 0;
    auto mm =
        MemoryManagerBuilder<TestCollectorScheme>().withConfig(config).build();
    REQUIRE(mm.getHeapSize() == config.initialMemory);
  }

  SECTION("Default initial memory") {
    auto mm = MemoryManagerBuilder<TestCollectorScheme>().build();
    REQUIRE(mm.getHeapSize() > 0);
  }

  SECTION("Other initial memory") {
    config.initialMemory = ab::mebibytes(8);
    auto mm =
        MemoryManagerBuilder<TestCollectorScheme>().withConfig(config).build();
    REQUIRE(mm.getHeapSize() >= config.initialMemory);
  }
}
