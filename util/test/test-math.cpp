#include <catch2/catch.hpp>
#include <omtalk/Util/Math.h>

using namespace omtalk;

TEST_CASE("ceilingDivide", "[math]") {
  REQUIRE(ceilingDivide(0, 2) == 0);
  REQUIRE(ceilingDivide(1, 2) == 1);
  REQUIRE(ceilingDivide(2, 2) == 1);
  REQUIRE(ceilingDivide(3, 2) == 2);
}
