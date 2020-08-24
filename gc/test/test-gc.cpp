#include "Object.h"
#include <catch2/catch.hpp>
#include <iostream>
#include <omtalk/Allocate.h>
#include <omtalk/Handle.h>
#include <omtalk/Heap.h>
#include <omtalk/MemoryManager.h>
#include <omtalk/Ref.h>
#include <omtalk/Tracing.h>
#include <omtalk/Util/BitArray.h>

//===----------------------------------------------------------------------===//
// TestSlotProxy
//===----------------------------------------------------------------------===//

class TestObjectProxy;

class TestSlotProxy {
public:
  TestSlotProxy(TestValue *target) : target(target) {}

  TestSlotProxy(const TestSlotProxy &) = default;

  template <omtalk::MemoryOrder M>
  gc::Ref<void> load() const noexcept {
    return omtalk::mem::load<M>(&target->asRef);
  }

  template <omtalk::MemoryOrder M, typename T>
  void store(gc::Ref<T> object) const noexcept {
    omtalk::mem::store<M>(&target->asRef,
                          object.template cast<TestObject>().get());
  }

private:
  TestValue *target;
};

//===----------------------------------------------------------------------===//
// TestObjectProxy
//===----------------------------------------------------------------------===//

template <typename C, typename V>
class SlotProxyVisitor {
public:
  explicit SlotProxyVisitor(V &visitor) : visitor(visitor) {}

  void visit(C &cx, TestValue *slot) { visitor.visit(TestSlotProxy(slot), cx); }
  V &visitor;
};

class TestObjectProxy {
public:
  explicit TestObjectProxy(gc::Ref<TestObject> obj) : target(obj) {}

  explicit TestObjectProxy(gc::Ref<TestStructObject> obj)
      : TestObjectProxy(obj.reinterpret<TestObject>()) {}

  explicit TestObjectProxy(gc::Ref<TestMapObject> obj)
      : TestObjectProxy(obj.reinterpret<TestObject>()) {}

  explicit TestObjectProxy(gc::Ref<void> obj)
      : TestObjectProxy(obj.reinterpret<TestObject>()) {}

  std::size_t getSize() const noexcept {
    switch (target->kind) {
    case TestObjectKind::STRUCT:
      return target.reinterpret<TestStructObject>()->getSize();
    case TestObjectKind::MAP:
      return target.reinterpret<TestMapObject>()->getSize();
    default:
      abort();
    }
  }

  std::size_t getForwardedSize() const noexcept { return getSize(); }

  bool isForwarded() const noexcept {
    return target->kind == TestObjectKind::INVALID;
  }

  gc::Ref<void> getForwardedAddress() const noexcept {
    assert(forwarded->kind == TestObjectKind::INVALID);
    auto forwarded = target.reinterpret<TestForwardedRecord>();
    return forwarded->to;
  }

  void forward(gc::Ref<void> to) noexcept {
    auto forwarded = target.reinterpret<TestForwardedRecord>();
    forwarded->kind = TestObjectKind::INVALID;
    forwarded->to = to.get();
  }

  template <typename ContextT, typename VisitorT>
  void walk(ContextT &cx, VisitorT &visitor) const noexcept {

    SlotProxyVisitor<ContextT, VisitorT> proxyVisitor(visitor);

    switch (target->kind) {
    case TestObjectKind::STRUCT:
      target.cast<TestStructObject>()->walk(cx, proxyVisitor);
      break;
    case TestObjectKind::MAP:
      target.cast<TestMapObject>()->walk(cx, proxyVisitor);
      break;
    default:
      abort();
    }
  }

  gc::Ref<TestObject> asRef() const noexcept { return target; }

private:
  gc::Ref<TestObject> target;
};

//===----------------------------------------------------------------------===//
// Test Collector
//===----------------------------------------------------------------------===//

struct TestCollectorScheme {
  using ObjectProxy = TestObjectProxy;
};

template <>
struct gc::GetProxy<TestCollectorScheme> {
  TestObjectProxy operator()(Ref<void> target) const noexcept {
    return TestObjectProxy(target.reinterpret<TestObject>());
  }
};

template <>
struct gc::RootWalker<TestCollectorScheme> {

  RootWalker() {}

  template <typename ContextT, typename VisitorT>
  void walk(ContextT &cx, VisitorT &visitor) noexcept {
    for (auto h : rootScope) {
      std::cout << "!!! rootwalker: handle " << h << std::endl;
      h->walk(visitor, cx);
    }
  }

  gc::RootHandleScope rootScope;
};

//===----------------------------------------------------------------------===//
// Test Allocator
//===----------------------------------------------------------------------===//

inline gc::Ref<TestStructObject>
allocateTestStructObject(gc::Context<TestCollectorScheme> &cx,
                         std::size_t nslots) noexcept {
  auto size = TestStructObject::allocSize(nslots);
  return gc::allocate<TestCollectorScheme, TestStructObject>(
      cx, size, [=](auto object) {
        object->kind = TestObjectKind::STRUCT;
        object->length = nslots;
        for (unsigned i = 0; i < nslots; i++) {
          object->setSlot(i, {REF, nullptr});
        }
      });
}

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

  auto *collector = &mm.getGlobalCollector();
  gc::Region *from = gc::Region::get(ref);
  gc::Region *to = mm.getRegionManager().allocateRegion();
  gc::GlobalCollectorContext<TestCollectorScheme> collectorContext(collector);

  collector->scanRoots(collectorContext);
  collector->completeScanning(collectorContext);
  gc::evacuate<TestCollectorScheme>(collectorContext, *from, *to);
}
