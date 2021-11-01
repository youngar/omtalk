#include <ab/Util/BitArray.h>
#include <catch2/catch.hpp>

using namespace ab;

static_assert(sizeof(BitChunkArray<1>) == sizeof(BitChunk) * 1);
static_assert(sizeof(BitChunkArray<4>) == sizeof(BitChunk) * 4);
static_assert(sizeof(BitChunkArray<8>) == sizeof(BitChunk) * 8);

static_assert(sizeof(BitChunk) == 4 || sizeof(BitChunk) == 8);
static_assert(isPow2(BITCHUNK_NBITS));

static_assert(sizeof(BitArray<BITCHUNK_NBITS>) == sizeof(BitChunk));
static_assert(sizeof(BitArray<BITCHUNK_NBITS * 4>) == sizeof(BitChunk) * 4);

TEST_CASE("zero length", "[BitArray]") {
  BitArray<0> bitArray;
  bitArray.clear();
}

TEST_CASE("n length", "[BitArray]") {
  BitArray<BITCHUNK_NBITS> bitArray;
  bitArray.clear();

  REQUIRE(bitArray.get(0) == false);
  REQUIRE(bitArray.get(1) == false);
  REQUIRE(bitArray.get(2) == false);

  REQUIRE(bitArray.set(1) == true);

  REQUIRE(bitArray.get(0) == false);
  REQUIRE(bitArray.get(1) == true);
  REQUIRE(bitArray.get(2) == false);

  REQUIRE(bitArray.unset(1) == true);

  REQUIRE(bitArray.get(0) == false);
  REQUIRE(bitArray.get(1) == false);
  REQUIRE(bitArray.get(2) == false);
}

TEST_CASE("multiple of n length", "[BitArray]") {
  BitArray<BITCHUNK_NBITS * 2> bitArray;
  bitArray.clear();

  REQUIRE(bitArray.get(BITCHUNK_NBITS + 0) == false);
  REQUIRE(bitArray.get(BITCHUNK_NBITS + 1) == false);
  REQUIRE(bitArray.get(BITCHUNK_NBITS + 2) == false);

  REQUIRE(bitArray.set(BITCHUNK_NBITS + 1) == true);

  REQUIRE(bitArray.get(BITCHUNK_NBITS + 0) == false);
  REQUIRE(bitArray.get(BITCHUNK_NBITS + 1) == true);
  REQUIRE(bitArray.get(BITCHUNK_NBITS + 2) == false);

  REQUIRE(bitArray.unset(BITCHUNK_NBITS + 1) == true);

  REQUIRE(bitArray.get(BITCHUNK_NBITS + 0) == false);
  REQUIRE(bitArray.get(BITCHUNK_NBITS + 1) == false);
  REQUIRE(bitArray.get(BITCHUNK_NBITS + 2) == false);
}

TEST_CASE("count", "[BitArray]") {
  BitArray<BITCHUNK_NBITS * 2> bitArray;
  bitArray.clear();
  REQUIRE(bitArray.count() == 0);
  REQUIRE(bitArray.set(BITCHUNK_NBITS + 1) == true);
  REQUIRE(bitArray.count() == 1);
  REQUIRE(bitArray.set(BITCHUNK_NBITS + 2) == true);
  REQUIRE(bitArray.count() == 2);
  bitArray.clear();
  REQUIRE(bitArray.count() == 0);
}