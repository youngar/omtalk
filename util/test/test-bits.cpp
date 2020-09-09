#include <catch2/catch.hpp>
#include <omtalk/Util/Bit.h>

using namespace omtalk;

TEST_CASE("poopcnt", "[bits]") {
    REQUIRE(popcount(0) == 0);
    REQUIRE(popcount(0b100101) == 3);
}
