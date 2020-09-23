#include <catch2/catch.hpp>
#include <example/Object.h>
#include <memory>
#include <omtalk/GlobalCollector.h>
#include <omtalk/Heap.h>
#include <omtalk/MemoryManager.h>
#include <omtalk/Ref.h>
#include <omtalk/Util/Bytes.h>

using namespace omtalk;
using namespace omtalk::gc;

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
    config.initialMemory = mebibytes(8);
    auto mm =
        MemoryManagerBuilder<TestCollectorScheme>().withConfig(config).build();
    REQUIRE(mm.getHeapSize() >= config.initialMemory);
  }
}
