#include <catch2/catch.hpp>
#include <omtalk/Util/Bit.h>

using namespace omtalk;

TEST_CASE("prefetch", "[bits]") {
    prefetch(nullptr);
}

TEST_CASE("popcount", "[bits]") {
    REQUIRE(popcount(0ul) == 0);
    REQUIRE(popcount(0b100101ul) == 3);
    REQUIRE(popcount(0b100000ul) == 1);
}

TEST_CASE("findFirstSet", "[bits]") {
    REQUIRE(findFirstSet(0l) == 0);
    REQUIRE(findFirstSet(0b100111l) == 1);
    REQUIRE(findFirstSet(0b100110l) == 2);
    REQUIRE(findFirstSet(0b100100l) == 3);
}

TEST_CASE("countLeadingZeros", "[bits]") {
    REQUIRE(countLeadingZeros(0ul) == 64);
    REQUIRE(countLeadingZeros(1ul) == 63);
    REQUIRE(countLeadingZeros(2ul) == 62);
    REQUIRE(countLeadingZeros(1ul << 63) == 0);
    REQUIRE(countLeadingZeros(1ul << 62) == 1);
}

TEST_CASE("countTrailingZeros", "[bits]") {
    REQUIRE(countTrailingZeros(0ul) == 64);
    REQUIRE(countTrailingZeros(1ul) == 0);
    REQUIRE(countTrailingZeros(2ul) == 1);
    REQUIRE(countTrailingZeros(3ul) == 0);
    REQUIRE(countTrailingZeros(4ul) == 2);
}

TEST_CASE("smear", "[bits]") {
    REQUIRE(smear(0b000) == 0b000);
    REQUIRE(smear(0b001) == 0b001);
    REQUIRE(smear(0b010) == 0b011);
    REQUIRE(smear(0b011) == 0b011);
    REQUIRE(smear(0b011) == 0b011);
    REQUIRE(smear(0b100) == 0b111);
    REQUIRE(smear(0b101) == 0b111);
    REQUIRE(smear(0b110) == 0b111);
    REQUIRE(smear(0b111) == 0b111);
}
