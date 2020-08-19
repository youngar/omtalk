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
// TestValueProxy
//===----------------------------------------------------------------------===//

class TestObjectProxy;

class TestValueProxy {
public:
  TestValueProxy(TestValue *target) : target(target) {}

  TestValueProxy(const TestValueProxy &) = default;

  gc::Ref<TestObject> loadRef() const noexcept { return target->asRef; }

  void storeRef(gc::Ref<TestObject> object) const noexcept {
    target->asRef = object.get();
  }

  TestObjectProxy loadProxy() const noexcept;

private:
  TestValue *target;
};

//===----------------------------------------------------------------------===//
// TestObjectProxy
//===----------------------------------------------------------------------===//

// template <typename S>
// struct SizeOf {
//   template <typename ObjectProxyT>
//   void operator()(Context &cx, ObjectProxyT target) const noexcept {
//     return target.size(cx);
//   }
// };

// template <typename C, typename V>
// void walk(C &cx, V &visitor) {
//   for (unsigned i = 0; i < length; i++) {
//     auto &slot = slots[i];
//     if (slot.kind == TestValue::Kind::REF) {
//       visitor.visit(cx, TestValueProxy(&slot));
//     }
//   }
// }

template <typename C, typename V>
class ValueProxyVisitor {
public:
  void visit(C &cx, TestValue *slot) {
    visitor.visit(cx, TestValueProxy(slot));
  }
  V &visitor;
};

class TestObjectProxy {
public:
  explicit TestObjectProxy(gc::Ref<TestObject> obj) : target(obj) {}

  explicit TestObjectProxy(gc::Ref<TestStructObject> obj)
      : TestObjectProxy(obj.reinterpret<TestObject>()) {}

  explicit TestObjectProxy(gc::Ref<TestMapObject> obj)
      : TestObjectProxy(obj.reinterpret<TestObject>()) {}

  std::size_t getSize() const noexcept {
    switch (target->kind) {
    case TestObjectKind::STRUCT:
      return target.reinterpret<TestStructObject>()->getSize();
    case TestObjectKind::MAP:
      return target.reinterpret<TestMapObject>()->getSize();
    default:
      return 0;
    }
  }

  template <typename ContextT, typename VisitorT>
  void walk(ContextT &cx, VisitorT &visitor) const noexcept {

    ValueProxyVisitor<ContextT, VisitorT> proxyVisitor(visitor);

    switch (target->kind) {
    case TestObjectKind::STRUCT:
      target.cast<TestStructObject>()->walk(cx, proxyVisitor);
      break;
    case TestObjectKind::MAP:
      target.cast<TestMapObject>()->walk(cx, proxyVisitor);
      break;
    default:
      break;
    }
  }

  gc::Ref<TestObject> get() const noexcept { return target; }

private:
  gc::Ref<TestObject> target;
};

//===----------------------------------------------------------------------===//
// TestValueProxy inlines
//===----------------------------------------------------------------------===//

inline TestObjectProxy TestValueProxy::loadProxy() const noexcept {
  return TestObjectProxy(target->asRef);
}

//===----------------------------------------------------------------------===//
// Test Collector
//===----------------------------------------------------------------------===//

struct TestCollectorScheme {
  using ObjectProxy = TestObjectProxy;
  using SlotProxy = TestValueProxy;
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
    // for (auto h : rootScope) {
    //   visitor.rootEdge(cx, RefProxy());
    // }
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
      cx, size, [=](auto object) { object->length = nslots; });
}

//===----------------------------------------------------------------------===//
// Test Startup and Shutdown
//===----------------------------------------------------------------------===//

TEST_CASE("Startup and Shutdown", "[garbage collector]") {
  auto mm = gc::MemoryManagerBuilder<TestCollectorScheme>()
                .withRootWalker(
                    std::make_unique<gc::RootWalker<TestCollectorScheme>>())
                .build();

  gc::Context<TestCollectorScheme> context(mm);
}

//===----------------------------------------------------------------------===//
// Test Exclusive Access
//===----------------------------------------------------------------------===//

TEST_CASE("Hanging", "[garbage collector]") {
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

TEST_CASE("Exclusive Access", "[garbage collector]") {
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

    // Prevent collect from starting
    std::unique_lock lock(mm.getYieldForGcMutex());

    // Start another thread to attempt to GC
    std::thread other([&]() { context2.collect(); });

    // Verify that a GC has not started
    REQUIRE(mm.getContextAccessCount() == 2);
    REQUIRE(context2.yieldForGC() == false);

    // Allow the GC to progress
    lock.unlock();

    // while (!mm.exclusiveRequested()) {
    // }

    while (!context.yieldForGC()) {
    }

    REQUIRE(context.yieldForGC() == false);

    std::thread another([&] {
      while (!context2.yieldForGC()) {
      }
    });

    context.collect();

    other.join();
    another.join();
  }

  REQUIRE(context.yieldForGC() == false);
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

    std::cout << "successful allocation\n" << *ref << std::endl;
  }
  return;
}