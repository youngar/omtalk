#include <catch2/catch.hpp>
#include <example/Object.h>
#include <memory>
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

TEST_CASE("Mark Roots", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  auto &gc = mm.getGlobalCollector();
  Context<TestCollectorScheme> context(mm);
  auto &gcContext = context.getCollectorContext();

  // get the region where the object was allocated
  HandleScope scope = context.getAuxData().rootScope.createScope();
  auto ref = allocateTestStructObject(context, 1);
  Handle<TestStructObject> handle(scope, ref);
  auto *region = Region::get(handle.get());

  // perform marking
  gc.preMark(gcContext);
  gc.markRoots(gcContext);
  gc.mark(gcContext);
  gc.postMark(gcContext);

  REQUIRE(region->marked(handle.get()));

  // second root
  ref = allocateTestStructObject(context, 1);
  Handle<TestStructObject> handle2(scope, ref);

  gc.preMark(gcContext);
  gc.markRoots(gcContext);
  gc.mark(gcContext);

  REQUIRE(region->marked(handle.get()));
  REQUIRE(region->marked(handle2.get()));
}

TEST_CASE("Mark Object Graph", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  auto &gc = mm.getGlobalCollector();
  Context<TestCollectorScheme> context(mm);
  auto &gcContext = context.getCollectorContext();

  // Allocate root
  HandleScope scope = context.getAuxData().rootScope.createScope();
  auto ref = allocateTestStructObject(context, 1);
  Handle<TestStructObject> root(scope, ref);

  // Allocate 10 object chain
  Handle<TestStructObject> handle = root;
  for (unsigned i = 0; i < 10; i++) {
    auto ref = allocateTestStructObject(context, 1);
    handle->setSlot(0, {REF, ref});
    handle = ref;
  }

  // perform marking
  gc.preMark(gcContext);
  gc.markRoots(gcContext);
  gc.mark(gcContext);

  // Verify by walking the chain
  handle = root;
  for (unsigned i = 0; i < 10; i++) {
    auto ref = handle.get();
    auto *region = Region::get(ref);
    REQUIRE(region->marked(ref));
    handle = makeRef(ref->getSlot(0).asRef).cast<TestStructObject>();
  }
}

TEST_CASE("Check live data", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  auto &gc = mm.getGlobalCollector();
  Context<TestCollectorScheme> context(mm);
  auto &gcContext = context.getCollectorContext();

  HandleScope scope = context.getAuxData().rootScope.createScope();
  unsigned nslots = 10;
  auto allocSize = TestStructObject::allocSize(nslots);

  // get the region where the object was allocated
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

TEST_CASE("Object Allocation", "[garbage collector]") {
  auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();
  auto &gc = mm.getGlobalCollector();
  Context<TestCollectorScheme> context(mm);
  auto &gcContext = context.getCollectorContext();

  HandleScope scope = context.getAuxData().rootScope.createScope();
  unsigned nslots = 10;
  auto allocSize = TestStructObject::allocSize(nslots);

  SECTION("Object is allocated white") {
    auto ref = allocateTestStructObject(context, nslots);
    auto *region = Region::get(ref);
    REQUIRE(region->unmarked(ref));
  }

  SECTION("Object is allocated black") {
    gc.preMark(gcContext);
    auto ref = allocateTestStructObject(context, nslots);
    auto *region = Region::get(ref);
    REQUIRE(region->marked(ref));
  }

  SECTION("Object allocated black during a gc cycle") {
    gc.preMark(gcContext);
    auto ref = allocateTestStructObject(context, nslots);
    auto *region = Region::get(ref);
    REQUIRE(region->marked(ref));
    gc.mark(gcContext);
    REQUIRE(region->getLiveObjectCount() == 1);
    REQUIRE(region->getLiveDataSize() == allocSize);
  }

  SECTION("Object is cleared on the cycle following the current cycle") {
    gc.preMark(gcContext);
    auto ref = allocateTestStructObject(context, nslots);
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