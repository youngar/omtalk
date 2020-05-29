#include <omtalk/Om/ObjectModel.h>

#include <catch2/catch.hpp>
#include <omtalk/Util/Atomic.h>

using namespace omtalk;

struct TestThingy {
  uintptr_t value1;
  uintptr_t value2;
};

namespace TestTypes {
enum {
  TestThingy,
  TestThingyArray
};
}

class TestType : public Type {
    // virtual 
};

class ThingyType : public TestType {


}

class ThingyArrayType : public TestType {

};


TEST_CASE("load", "[object-model]") {

  auto x = int(1234);
  REQUIRE(atomic::load(&x) == 1234);
}

TEST_CASE("store then load", "[atomic]") {
  auto x = int(1234);
  REQUIRE(x == 1234);

  atomic::store(&x, 5678);
  REQUIRE(x == 5678);
  REQUIRE(atomic::load(&x) == 5678);
}

TEST_CASE("exchange", "[atomic]") {
  int x = 1234;
  REQUIRE(x == 1234);

  auto y = atomic::exchange(&x, 5678);
  REQUIRE(x == 5678);
  REQUIRE(y == 1234);
  REQUIRE(atomic::load(&x) == 5678);
  REQUIRE(atomic::load(&y) == 1234);
}

SCENARIO("integers can be compare/exchanged", "[atomic]") {
  GIVEN("an integer 1234") {
    int x = 1234;
    REQUIRE(1234 == x);

    WHEN("we try to compare/exchange with the wrong expected value") {
      auto succ = atomic::compareExchange(&x, 5678, 9012);
      THEN("the value should remain unchanged") {
        REQUIRE(!succ);
        REQUIRE(1234 == x);
        REQUIRE(1234 == atomic::load(&x));
      }
    }

    WHEN("we compare/exchange with the correct expected value") {
      auto succ = atomic::compareExchange(&x, 1234, 5678);
      THEN("the value should change") {
        REQUIRE(succ);
        REQUIRE(5678 == x);
        REQUIRE(5678 == atomic::load(&x));
      }
    }
  }
}
