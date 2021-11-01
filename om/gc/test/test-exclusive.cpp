#include <ab/Util/Atomic.h>
#include <ab/Util/IntrusiveList.h>
#include <catch2/catch.hpp>
#include <example/Object.h>
#include <memory>
#include <om/GC/Handle.h>
#include <om/GC/Heap.h>
#include <om/GC/MemoryManager.h>
#include <om/GC/Ref.h>
#include <thread>

using namespace om;
using namespace om::gc;

TEST_CASE("Exclusive requested check", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  Context<TestCollectorScheme> context(mm);

  Context<TestCollectorScheme> context2(mm);
  std::thread other([&]() { context2.collect(); });

  while (!mm.exclusiveRequested()) {
  }
  REQUIRE(context.yieldForGC() == true);
  other.join();
}

TEST_CASE("Exclusive Access blocked by other thread", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  Context<TestCollectorScheme> context(mm);
  REQUIRE(mm.getContextAccessCount() == 1);
  REQUIRE(context.yieldForGC() == false);
  context.collect();

  Context<TestCollectorScheme> context2(mm);
  std::thread other([&] { context2.collect(); });

  while (!mm.exclusiveRequested()) {
    // spin
  }

  while (mm.getContextAccessCount() == 2) {
    // spin
  }

  REQUIRE(mm.getContextCount() == 2);
  REQUIRE(mm.getContextAccessCount() == 1);

  REQUIRE(context.yieldForGC() == true);

  REQUIRE(mm.getContextCount() == 2);
  REQUIRE(mm.getContextAccessCount() == 2);

  REQUIRE(context.yieldForGC() == false);

  REQUIRE(mm.getContextCount() == 2);
  REQUIRE(mm.getContextAccessCount() == 2);

  other.join();
}

TEST_CASE("Exclusive access not blocked by destroyed context",
          "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  Context<TestCollectorScheme> context(mm);
  REQUIRE(mm.getContextAccessCount() == 1);
  REQUIRE(context.yieldForGC() == false);
  context.collect();

  {
    Context<TestCollectorScheme> context2(mm);
    REQUIRE(mm.getContextAccessCount() == 2);
    REQUIRE(context.yieldForGC() == false);
    REQUIRE(context2.yieldForGC() == false);
  }

  context.collect();
  REQUIRE(context.yieldForGC() == false);
}
