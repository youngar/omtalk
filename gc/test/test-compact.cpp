#include <catch2/catch.hpp>
#include <example/Object.h>
#include <iostream>
#include <memory>
#include <omtalk/Barrier.h>
#include <omtalk/GlobalCollector.h>
#include <omtalk/Handle.h>
#include <omtalk/Heap.h>
#include <omtalk/MemoryManager.h>
#include <omtalk/Ref.h>

namespace omtalk::gc {
template <typename S>
struct RootWalker;
} // namespace omtalk::gc

using namespace omtalk::gc;

TEST_CASE("Compact Root Fixup", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  auto &gc = mm.getGlobalCollector();
  Context<TestCollectorScheme> context(mm);
  auto &gcContext = context.getCollectorContext();

  // get the region where the object was allocated
  
  HandleScope scope = context.getAuxData().rootScope.createScope();
  auto ref = allocateTestStructObject(context, 3);
  ref->setSlot(0, {INT, 4});
  ref->setSlot(1, {INT, 5});
  ref->setSlot(2, {INT, 6});

  Handle<TestStructObject> handle(scope, ref);

  auto *region = Region::get(handle.get());

  // return the allocation region to be evacuated
  context.refreshBuffer(0);
  gc.collect(gcContext);
  gc.collect(gcContext);

  // make sure the object was actually moved
  CHECK(ref != handle.get());
  CHECK(region->isEvacuating());

  // object should still be valid
  CHECK(handle->getSlot(0) == TestValue(INT, 4));
  CHECK(handle->getSlot(1) == TestValue(INT, 5));
  CHECK(handle->getSlot(2) == TestValue(INT, 6));
}

TEST_CASE("Compact Load Barrier", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  auto &gc = mm.getGlobalCollector();
  Context<TestCollectorScheme> context(mm);
  auto &gcContext = context.getCollectorContext();

  HandleScope scope = context.getAuxData().rootScope.createScope();
  auto ref = allocateTestStructObject(context, 1);
  Handle<TestStructObject> handle(scope, ref);

  ref = allocateTestStructObject(context, 1);
  handle->setSlot(0, {REF, ref});

  std::cout << "\n\nGOOD TEST\n";
  context.refreshBuffer(0);

  //   gc.collect(gcContext);
  gc.preMark(gcContext);
  gc.markRoots(gcContext);
  gc.mark(gcContext);
  gc.postMark(gcContext);
  gc.preCompact(gcContext);
  // stop before the object is copied

  ref = handle.get();
  load(context, handle);
  ref = handle.get();
}
