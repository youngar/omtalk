#include <ab/Util/Atomic.h>
#include <ab/Util/IntrusiveList.h>
#include <catch2/catch.hpp>
#include <cstddef>
#include <example/Object.h>
#include <memory>
#include <om/GC/GlobalCollector.h>
#include <om/GC/Handle.h>
#include <om/GC/Heap.h>
#include <om/GC/MemoryManager.h>
#include <om/GC/Ref.h>

using namespace om;
using namespace om::gc;

namespace om::gc {
template <typename S>
struct RootWalker;
} // namespace om::gc

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
  HandleScope scope = context.getAuxData().rootScope.createScope();
  auto ref = allocateTestStructObject(context, 10);
  Handle<TestStructObject> handle(scope, ref);
}

TEST_CASE("Allocate Cache Flushing", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  Context<TestCollectorScheme> context(mm);
  HandleScope scope = context.getAuxData().rootScope.createScope();
  auto allocSize = TestStructObject::allocSize(10);
  auto ref = allocateTestStructObject(context, 10);
  auto *region = Region::get(ref);
  // refresh the buffer and make sure that all region statistics are up to date
  context.refreshBuffer(1);
  auto *freeAddress = (std::byte *)(ref.toAddr() + allocSize);
  CHECK(region->getFree() == freeAddress);
  // CHECK(region->getLiveDataSize() == allocSize);
  // CHECK(region->getLiveObjectCount() == 1);
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
  HandleScope scope = context.getAuxData().rootScope.createScope();
  auto ref = allocateTestStructObject(context, 10);
  Handle<TestStructObject> handle(scope, ref);
  mm.collect(context);
}

TEST_CASE("Allocate object tree and gc", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  Context<TestCollectorScheme> context(mm);
  HandleScope scope = context.getAuxData().rootScope.createScope();
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

  Context<TestCollectorScheme> context(mm);
  HandleScope scope = context.getAuxData().rootScope.createScope();

  for (int i = 0; i < 1; i++) {
    auto ref = allocateTestStructObject(context, 10);
    Handle<TestStructObject> handle(scope, ref);
    REQUIRE(ref != nullptr);
  }
  return;
}

TEST_CASE("Concurrent", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  auto &gc = mm.getGlobalCollector();
  Context<TestCollectorScheme> context(mm);
  auto &gcContext = context.getCollectorContext();

  HandleScope scope = context.getAuxData().rootScope.createScope();
  unsigned nslots = 10;
  auto ref = allocateTestStructObject(context, nslots);
  Handle<TestStructObject> handle(scope, ref);

  mm.kickoff(context);
  gc.wait(gcContext);
}
