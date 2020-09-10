#include "Object.h"
#include <catch2/catch.hpp>
#include <iostream>
#include <memory>
#include <omtalk/GlobalCollector.h>
#include <omtalk/Handle.h>
#include <omtalk/Heap.h>
#include <omtalk/MemoryManager.h>
#include <omtalk/Ref.h>
#include <thread>

namespace omtalk::gc {
template <typename S>
struct RootWalker;
} // namespace omtalk::gc

//===----------------------------------------------------------------------===//
// Test Startup and Shutdown
//===----------------------------------------------------------------------===//

TEST_CASE("Startup and Shutdown MemoryManager", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();
}

TEST_CASE("Startup and Shutdown MemoryManager and Contexts",
          "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  gc::Context<TestCollectorScheme> context1(mm);
  gc::Context<TestCollectorScheme> context2(mm);
  gc::Context<TestCollectorScheme> context3(mm);
}

//===----------------------------------------------------------------------===//
// Test Exclusive Access
//===----------------------------------------------------------------------===//

TEST_CASE("Exclusive requested check", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  gc::Context<TestCollectorScheme> context(mm);

  gc::Context<TestCollectorScheme> context2(mm);
  std::thread other([&]() { context2.collect(); });

  while (!mm.exclusiveRequested()) {
  }
  REQUIRE(context.yieldForGC() == true);
  other.join();
}

TEST_CASE("Exclusive Access blocked by other thread", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  gc::Context<TestCollectorScheme> context(mm);
  REQUIRE(mm.getContextAccessCount() == 1);
  REQUIRE(context.yieldForGC() == false);
  context.collect();

  gc::Context<TestCollectorScheme> context2(mm);
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
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  gc::Context<TestCollectorScheme> context(mm);
  REQUIRE(mm.getContextAccessCount() == 1);
  REQUIRE(context.yieldForGC() == false);
  context.collect();

  {
    gc::Context<TestCollectorScheme> context2(mm);
    REQUIRE(mm.getContextAccessCount() == 2);
    REQUIRE(context.yieldForGC() == false);
    REQUIRE(context2.yieldForGC() == false);
  }

  context.collect();
  REQUIRE(context.yieldForGC() == false);
}

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

TEST_CASE("Allocate single garbage", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  gc::Context<TestCollectorScheme> context(mm);

  allocateTestStructObject(context, 10);
}

TEST_CASE("Allocate single root", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  gc::Context<TestCollectorScheme> context(mm);

  gc::HandleScope scope = mm.getRootWalker().rootScope.createScope();

  auto ref = allocateTestStructObject(context, 10);
  gc::Handle<TestStructObject> handle(scope, ref);
}

//===----------------------------------------------------------------------===//
// Garbage Collection
//===----------------------------------------------------------------------===//

TEST_CASE("Allocate single garbage and gc", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  gc::Context<TestCollectorScheme> context(mm);

  allocateTestStructObject(context, 10);

  mm.collect(context);
}

TEST_CASE("Allocate single root and gc", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();
  gc::Context<TestCollectorScheme> context(mm);
  gc::HandleScope scope = mm.getRootWalker().rootScope.createScope();
  auto ref = allocateTestStructObject(context, 10);
  gc::Handle<TestStructObject> handle(scope, ref);
  mm.collect(context);
}

TEST_CASE("Allocate object tree and gc", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  std::cout << "!!! Start of the good test" << std::endl;

  gc::Context<TestCollectorScheme> context(mm);
  gc::HandleScope scope = mm.getRootWalker().rootScope.createScope();

  std::cout << "test: handleCount:"
            << mm.getRootWalker().rootScope.handleCount() << std::endl;

  gc::Handle<TestStructObject> handle(scope,
                                      allocateTestStructObject(context, 10));

  auto ref = allocateTestStructObject(context, 10);
  ref = allocateTestStructObject(context, 10);
  handle->setSlot(0, {REF, ref});

  mm.collect(context);
}

TEST_CASE("Root scanning", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  gc::HandleScope scope = mm.getRootWalker().rootScope.createScope();

  gc::Context<TestCollectorScheme> context(mm);

  for (int i = 0; i < 1; i++) {

    auto ref = allocateTestStructObject(context, 10);
    gc::Handle<TestStructObject> handle(scope, ref);

    if (ref == nullptr) {
      std::cout << "Bad Allocation\n";
      break;
    }
  }
  return;
}

TEST_CASE("Check live data", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();
  gc::Context<TestCollectorScheme> context(mm);
  gc::HandleScope scope = mm.getRootWalker().rootScope.createScope();
  unsigned nslots = 10;
  auto allocSize = TestStructObject::allocSize(nslots);
  auto ref = allocateTestStructObject(context, nslots);
  gc::Handle<TestStructObject> handle(scope, ref);

  auto &gc = mm.getGlobalCollector();
  gc::GlobalCollectorContext<TestCollectorScheme> gcContext(&gc);

  auto *region = gc::Region::get(ref);

  // perform marking
  gc.setup(gcContext);
  gc.scanRoots(gcContext);
  gc.completeScanning(gcContext);

  // Check live data
  REQUIRE(region->getLiveDataSize() == allocSize);

  // Do it all again! makes sure things are properly reset between GCs
  gc.setup(gcContext);
  gc.scanRoots(gcContext);
  gc.completeScanning(gcContext);

  // Check live data
  REQUIRE(region->getLiveDataSize() == allocSize);
}

TEST_CASE("Region selection", "[garbage collector]") {

}


//===----------------------------------------------------------------------===//
// Evacuation
//===----------------------------------------------------------------------===//

TEST_CASE("Evacuation", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  gc::Context<TestCollectorScheme> context(mm);

  gc::HandleScope scope = mm.getRootWalker().rootScope.createScope();

  gc::Handle<TestStructObject> handle(scope,
                                      allocateTestStructObject(context, 10));

  auto ref = allocateTestStructObject(context, 10);
  ref = allocateTestStructObject(context, 10);
  handle->setSlot(0, {REF, ref});

  std::cout << "object " << handle.get() << std::endl;
  std::cout << "object " << ref << std::endl;

  // auto *collector = &mm.getGlobalCollector();
  // gc::Region *from = gc::Region::get(ref);
  // gc::Region *to = mm.getRegionManager().allocateRegion();
  // gc::GlobalCollectorContext<TestCollectorScheme>
  // collectorContext(collector);

  // collector->scanRoots(collectorContext);
  // collector->completeScanning(collectorContext);
  // gc::evacuate<TestCollectorScheme>(collectorContext, *from, *to);
}
