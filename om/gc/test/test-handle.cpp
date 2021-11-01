#include <ab/Util/Atomic.h>
#include <catch2/catch.hpp>
#include <cstdint>
#include <om/GC/Handle.h>
#include <om/GC/Ref.h>

using namespace om;
using namespace om::gc;

TEST_CASE("EmptyScope", "[garbage collector]") {
  RootHandleScope rootScope;
  REQUIRE(rootScope.handleCount() == 0);
}

TEST_CASE("EmptyInnerScope", "[garbage collector]") {
  RootHandleScope rootScope;
  REQUIRE(rootScope.handleCount() == 0);
  {
    HandleScope inner = rootScope.createScope();
    REQUIRE(rootScope.handleCount() == 0);
    REQUIRE(inner.handleCount() == 0);
  }
  REQUIRE(rootScope.handleCount() == 0);
}

TEST_CASE("RootScope", "[garbage collector]") {
  RootHandleScope rootScope;
  uintptr_t testValue;
  Handle<std::uintptr_t> handle1(rootScope, Ref<std::uintptr_t>(&testValue));
  REQUIRE(rootScope.handleCount() == 1);
  Handle<std::uintptr_t> handle2(rootScope, Ref<std::uintptr_t>(&testValue));
  REQUIRE(rootScope.handleCount() == 2);
}

TEST_CASE("InnerScope", "[garbage collector]") {
  uintptr_t testValue;

  RootHandleScope rootScope;
  {
    HandleScope inner = rootScope.createScope();
    Handle<std::uintptr_t> handle(inner, Ref<std::uintptr_t>(&testValue));
    REQUIRE(rootScope.handleCount() == 1);
    REQUIRE(inner.handleCount() == 1);
  }
  REQUIRE(rootScope.handleCount() == 0);
}

TEST_CASE("WalkHandles", "[garbage collector]") {
  uintptr_t testValue = 100;

  RootHandleScope rootScope;
  Handle<std::uintptr_t> handle(rootScope, Ref<std::uintptr_t>(&testValue));
  {
    HandleScope inner = rootScope.createScope();
    Handle<std::uintptr_t> handle(inner, Ref<std::uintptr_t>(&testValue));
    unsigned count = 0;
    for (auto *n : rootScope) {
      count++;
      Handle<std::uintptr_t> *h = static_cast<Handle<std::uintptr_t> *>(n);
      REQUIRE(*h->load<ab::RELAXED>().reinterpret<uintptr_t>() == 100);
    }
    REQUIRE(count == 2);
  }
  unsigned count = 0;
  for (auto *n : rootScope) {
    count++;
    REQUIRE(*n->load<ab::RELAXED>().reinterpret<uintptr_t>() == 100);
  }
  REQUIRE(count == 1);
}

TEST_CASE("Handle Assignment", "[garbage collector]") {
  RootHandleScope rootScope;

  std::uintptr_t value1 = 5;
  Ref<std::uintptr_t> ref1 = &value1;
  Handle<std::uintptr_t> handle1(rootScope, ref1);

  std::uintptr_t value2 = 6;
  Ref<std::uintptr_t> ref2 = &value2;
  Handle<std::uintptr_t> handle2(rootScope, ref2);

  CHECK(*handle1 == value1);
  CHECK(*handle2 == value2);

  handle1 = handle2;

  CHECK(*handle1 == value2);
  CHECK(*handle2 == value2);

  handle1 = ref1;

  CHECK(*handle1 == value1);
  CHECK(*handle2 == value2);
}
