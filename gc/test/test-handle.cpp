#include <catch2/catch.hpp>
#include <cstdint>
#include <omtalk/Handle.h>
#include <omtalk/Ref.h>
#include <omtalk/Util/Atomic.h>

using namespace omtalk;
using namespace omtalk::gc;

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
      REQUIRE(*h->load<RELAXED>().reinterpret<uintptr_t>() == 100);
    }
    REQUIRE(count == 2);
  }
  unsigned count = 0;
  for (auto *n : rootScope) {
    count++;
    REQUIRE(*n->load<RELAXED>().reinterpret<uintptr_t>() == 100);
  }
  REQUIRE(count == 1);
}
