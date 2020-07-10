#include <omtalk/Om/ObjectModel.h>

#include <catch2/catch.hpp>
#include <omtalk/Util/Atomic.h>

using namespace omtalk;

struct TestThingy {
  uintptr_t value1;
  uintptr_t value2;
};

namespace TestTypes {
enum { TestThingy, TestThingyArray };
}

class TestType : public Type {};

class ThingyType : public TestType {};

class ThingyArrayType : public TestType {};
