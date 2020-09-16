#ifndef OMTALK_GC_TEST_OBJECT_H_
#define OMTALK_GC_TEST_OBJECT_H_

#include <cassert>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <omtalk/Allocate.h>
#include <omtalk/GlobalCollector.h>
#include <omtalk/Handle.h>
#include <omtalk/Heap.h>
#include <omtalk/MemoryManager.h>
#include <omtalk/Ref.h>
#include <omtalk/Scheme.h>
#include <omtalk/Util/Atomic.h>
#include <ostream>
#include <thread>

struct TestObject;
namespace gc = omtalk::gc;

//===----------------------------------------------------------------------===//
// TestValue
//===----------------------------------------------------------------------===//

struct RefTag {};
struct IntTag {};
constexpr RefTag REF;
constexpr IntTag INT;

struct TestValue {
  enum class Kind { REF, INT };

  TestValue() {}
  TestValue(IntTag, int x) : asInt(x), kind(Kind::INT) {}
  TestValue(RefTag, TestObject *x) : asRef(x), kind(Kind::REF) {}
  TestValue(RefTag, gc::Ref<TestObject> x) : asRef(x.get()), kind(Kind::REF) {}

  union {
    TestObject *asRef;
    int asInt;
  };
  Kind kind;
};

std::ostream &operator<<(std::ostream &out, const TestValue::Kind &kind) {
  switch (kind) {
  case TestValue::Kind::REF:
    out << "REF";
    break;
  case TestValue::Kind::INT:
    out << "INT";
    break;
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const TestValue &obj) {
  out << "(TestValue kind: " << obj.kind << ", value: ";
  if (obj.kind == TestValue::Kind::REF) {
    out << obj.asRef;
  } else {
    out << obj.asInt;
  }
  out << ")";
  return out;
}

//===----------------------------------------------------------------------===//
// TestObjectKind
//===----------------------------------------------------------------------===//

enum class TestObjectKind { INVALID, STRUCT, MAP };

std::ostream &operator<<(std::ostream &out, const TestObjectKind &obj) {
  switch (obj) {
  case TestObjectKind::INVALID:
    out << "INVALID";
    break;
  case TestObjectKind::STRUCT:
    out << "STRUCT";
    break;
  case TestObjectKind::MAP:
    out << "MAP";
    break;
  }
  return out;
}

//===----------------------------------------------------------------------===//
// TestObject
//===----------------------------------------------------------------------===//

struct TestObject {
  TestObjectKind kind;
};

std::ostream &operator<<(std::ostream &out, const TestObject &obj) {
  out << "(TestObject";
  out << " kind: " << obj.kind;
  out << ")";
  return out;
}

struct TestForwardedRecord : TestObject {
  void *to;
};

//===----------------------------------------------------------------------===//
// TestStructObject
//===----------------------------------------------------------------------===//

struct TestStructObject : public TestObject {
  static constexpr std::size_t allocSize(std::size_t nslots) {
    return sizeof(TestStructObject) + (sizeof(TestValue) * nslots);
  }

  std::size_t getSize() const noexcept { return allocSize(length); }

  std::size_t getLength() const noexcept { return length; }

  void setSlot(unsigned slot, TestValue value) noexcept { slots[slot] = value; }

  template <typename C, typename V>
  void walk(C &cx, V &visitor) {
    std::cout << "!!! TestStructObject::walk: " << this << std::endl;
    for (unsigned i = 0; i < length; i++) {
      auto &slot = slots[i];
      if (slot.kind == TestValue::Kind::REF && slot.asRef != nullptr) {
        std::cout << "!!!   slot:" << slot << std::endl;
        visitor.visit(cx, &slot);
      }
    }
  }

  std::size_t length;
  TestValue slots[];
};

std::ostream &operator<<(std::ostream &out, const TestStructObject &obj) {
  out << "(TestStructObject";
  out << " kind: " << obj.kind;
  out << ", length: " << obj.length;
  out << ",\n";
  for (unsigned i = 0; i < obj.getLength(); i++)
    out << "  slot[" << i << "]: " << obj.slots[i] << std::endl;
  out << ")";
  return out;
}

//===----------------------------------------------------------------------===//
// TestMapObject
//===----------------------------------------------------------------------===//

///
/// TODO
///

struct TestMapObject : public TestObject {

  struct Bucket {
    int key;
    TestValue value;
  };

  static constexpr std::size_t allocSize(std::size_t nslots) {
    return sizeof(TestMapObject) + (sizeof(Bucket) * nslots);
  }

  static constexpr int TOMBSTONE = std::numeric_limits<int>::max();

  std::size_t getSize() const noexcept { return allocSize(length); }

  std::size_t getLength() const noexcept { return length; }

  bool insert(int key, TestValue value) noexcept { return false; }

  bool remove(int key) noexcept { return false; }

  TestValue get(int key) const noexcept { return {INT, 0}; }

  template <typename C, typename V>
  void walk(C &cx, V &visitor) {
    for (unsigned i = 0; i < length; i++) {
      auto &slot = buckets[i].value;
      if (slot.kind == TestValue::Kind::REF) {
        visitor.visit(cx, &slot);
      }
    }
  }

  std::size_t length;
  Bucket buckets[];
};

std::ostream &operator<<(std::ostream &out,
                         const TestMapObject::Bucket &bucket) {
  out << "key: " << bucket.key;
  out << " value: " << bucket.value;
  return out;
}

std::ostream &operator<<(std::ostream &out, const TestMapObject &obj) {
  out << "(TestMapObject";
  out << " kind: " << obj.kind;
  out << " length: " << obj.length;
  for (unsigned i = 0; i < obj.getLength(); i++)
    out << " bucket[" << i << "]: " << obj.buckets[i];
  out << ")";
  return out;
}

namespace omtalk {
namespace gc {
template <typename S>
struct GetProxy;

template <typename S>
struct RootWalker;
} // namespace gc
} // namespace omtalk

//===----------------------------------------------------------------------===//
// TestSlotProxy
//===----------------------------------------------------------------------===//

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

  constexpr operator void *() const noexcept { return target.get(); }

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
    for (auto *h : rootScope) {
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

#endif