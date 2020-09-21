#include "Object.h"
#include <catch2/catch.hpp>
#include <iostream>
#include <memory>
#include <omtalk/GlobalCollector.h>
#include <omtalk/Handle.h>
#include <omtalk/Heap.h>
#include <omtalk/MemoryManager.h>
#include <omtalk/Ref.h>

using namespace omtalk;
using namespace omtalk::gc;

namespace omtalk::gc {
template <typename S>
struct RootWalker;
} // namespace omtalk::gc

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

TEST_CASE("Allocate single garbage", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  Context<TestCollectorScheme> context(mm);

  allocateTestStructObject(context, 10);
}

TEST_CASE("Allocate single root", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  Context<TestCollectorScheme> context(mm);
  HandleScope scope = mm.getRootWalker().rootScope.createScope();
  auto ref = allocateTestStructObject(context, 10);
  Handle<TestStructObject> handle(scope, ref);
}

//===----------------------------------------------------------------------===//
// Garbage Collection
//===----------------------------------------------------------------------===//

TEST_CASE("Allocate single garbage and gc", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  Context<TestCollectorScheme> context(mm);
  allocateTestStructObject(context, 10);
  mm.collect(context);
}

TEST_CASE("Allocate single root and gc", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  Context<TestCollectorScheme> context(mm);
  HandleScope scope = mm.getRootWalker().rootScope.createScope();
  auto ref = allocateTestStructObject(context, 10);
  Handle<TestStructObject> handle(scope, ref);
  mm.collect(context);
}

TEST_CASE("Allocate object tree and gc", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  std::cout << "!!! Start of the good test" << std::endl;

  Context<TestCollectorScheme> context(mm);
  HandleScope scope = mm.getRootWalker().rootScope.createScope();

  std::cout << "test: handleCount:"
            << mm.getRootWalker().rootScope.handleCount() << std::endl;

  Handle<TestStructObject> handle(scope, allocateTestStructObject(context, 10));

  auto ref = allocateTestStructObject(context, 10);
  ref = allocateTestStructObject(context, 10);
  handle->setSlot(0, {REF, ref});

  mm.collect(context);
}

TEST_CASE("Root scanning", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  HandleScope scope = mm.getRootWalker().rootScope.createScope();

  Context<TestCollectorScheme> context(mm);

  for (int i = 0; i < 1; i++) {

    auto ref = allocateTestStructObject(context, 10);
    Handle<TestStructObject> handle(scope, ref);

    if (ref == nullptr) {
      std::cout << "Bad Allocation\n";
      break;
    }
  }
  return;
}

TEST_CASE("Check live data", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  auto &gc = mm.getGlobalCollector();
  Context<TestCollectorScheme> context(mm);
  auto &gcContext = context.getCollectorContext();

  HandleScope scope = mm.getRootWalker().rootScope.createScope();
  unsigned nslots = 10;
  auto allocSize = TestStructObject::allocSize(nslots);

  // get the region where the object was allocated

  SECTION("Marking sets the proper live data size in a region") {
    auto ref = allocateTestStructObject(context, nslots);
    Handle<TestStructObject> handle(scope, ref);
    auto *region = Region::get(handle.get());

    // perform marking
    gc.preMark(gcContext);
    gc.markRoots(gcContext);
    gc.mark(gcContext);

    // Check live data
    REQUIRE(region->getLiveObjectCount() == 1);
    REQUIRE(region->getLiveDataSize() == allocSize);

    // Do it all again! makes sure things are properly reset between GCs
    gc.preMark(gcContext);
    gc.markRoots(gcContext);
    gc.mark(gcContext);

    // Check live data
    REQUIRE(region->getLiveObjectCount() == 1);
    REQUIRE(region->getLiveDataSize() == allocSize);
  }

  SECTION("Object is allocated white") {
    auto ref = allocateTestStructObject(context, 10);
    auto *region = Region::get(ref);
    REQUIRE(region->unmarked(ref));
  }

  SECTION("Object is allocated black") {
    // premark enables black allocation
    gc.preMark(gcContext);
    auto ref = allocateTestStructObject(context, 10);
    auto *region = Region::get(ref);
    REQUIRE(region->marked(ref));
  }

  SECTION("Object allocated black during a gc cycle") {
    gc.preMark(gcContext);
    auto ref = allocateTestStructObject(context, 10);
    auto *region = Region::get(ref);
    REQUIRE(region->marked(ref));
    gc.mark(gcContext);
    REQUIRE(region->getLiveObjectCount() == 1);
    REQUIRE(region->getLiveDataSize() == allocSize);
  }

  SECTION("Object is cleared on the cycle following the current cycle") {
    gc.preMark(gcContext);
    auto ref = allocateTestStructObject(context, 10);
    auto *region = Region::get(ref);
    CHECK(region->marked(ref));
    gc.mark(gcContext);
    CHECK(region->marked(ref));
    CHECK(region->getLiveObjectCount() == 1);
    CHECK(region->getLiveDataSize() == allocSize);
    gc.collect(gcContext);
    CHECK(region->unmarked(ref));
    CHECK(region->getLiveObjectCount() == 0);
    CHECK(region->getLiveDataSize() == 0);
  }
}

TEST_CASE("Concurrent", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  auto &gc = mm.getGlobalCollector();
  Context<TestCollectorScheme> context(mm);
  auto &gcContext = context.getCollectorContext();

  HandleScope scope = mm.getRootWalker().rootScope.createScope();
  unsigned nslots = 10;
  // auto allocSize = TestStructObject::allocSize(nslots);
  auto ref = allocateTestStructObject(context, nslots);
  Handle<TestStructObject> handle(scope, ref);

  mm.kickoff(context);
  gc.wait(gcContext);
}

//===----------------------------------------------------------------------===//
// Evacuation
//===----------------------------------------------------------------------===//

TEST_CASE("Evacuation", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  Context<TestCollectorScheme> context(mm);

  HandleScope scope = mm.getRootWalker().rootScope.createScope();

  Handle<TestStructObject> handle(scope, allocateTestStructObject(context, 10));

  auto ref = allocateTestStructObject(context, 10);
  ref = allocateTestStructObject(context, 10);
  handle->setSlot(0, {REF, ref});

  std::cout << "object " << handle.get() << std::endl;
  std::cout << "object " << ref << std::endl;

  // auto *collector = &mm.getGlobalCollector();
  // Region *from = Region::get(ref);
  // Region *to = mm.getRegionManager().allocateRegion();
  // GlobalCollectorContext<TestCollectorScheme>
  // collectorContext(collector);

  // collector->scanRoots(collectorContext);
  // collector->completeScanning(collectorContext);
  // evacuate<TestCollectorScheme>(collectorContext, *from, *to);
}
