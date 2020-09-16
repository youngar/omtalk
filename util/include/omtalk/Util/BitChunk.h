#ifndef OMTALK_UTIL_BITCHUNK_H_
#define OMTALK_UTIL_BITCHUNK_H_

#include <array>
#include <cstdint>
#include <omtalk/Util/Bytes.h>

namespace omtalk {

enum class BitChunk : std::uintptr_t {};

constexpr BitChunk BITCHUNK_ALL = BitChunk(UINTPTR_MAX);
constexpr BitChunk BITCHUNK_NONE = BitChunk(0);

constexpr std::size_t BITCHUNK_NBITS = sizeof(BitChunk) * 8;

constexpr BitChunk operator<<(BitChunk chunk, std::size_t shift) {
  return BitChunk(std::uintptr_t(chunk) << shift);
}

constexpr BitChunk operator|(BitChunk lhs, BitChunk rhs) {
  return BitChunk(std::uintptr_t(lhs) | std::uintptr_t(rhs));
}

constexpr BitChunk operator&(BitChunk lhs, BitChunk rhs) {
  return BitChunk(std::uintptr_t(lhs) & std::uintptr_t(rhs));
}

constexpr BitChunk operator~(BitChunk chunk) {
  return BitChunk(~std::uintptr_t(chunk));
}

constexpr BitChunk &operator|=(BitChunk &lhs, BitChunk rhs) {
  return lhs = lhs | rhs;
}

constexpr BitChunk &operator&=(BitChunk &lhs, BitChunk rhs) {
  return lhs = lhs & rhs;
}

constexpr bool toBool(BitChunk chunk) {
  return chunk != BitChunk(0);
}

template <std::size_t N>
using BitChunkArray = std::array<BitChunk, N>;

static constexpr std::size_t indexForBit(std::size_t index) {
  return index / BITCHUNK_NBITS;
}

static constexpr std::size_t shiftForBit(std::size_t index) {
  return index % BITCHUNK_NBITS;
}

static constexpr BitChunk maskForBit(std::size_t index) {
  return BitChunk(1) << shiftForBit(index);
}

constexpr bool all(BitChunk chunk) {
  return chunk == BITCHUNK_ALL;
}

constexpr bool any(BitChunk chunk) {
  return chunk != BITCHUNK_NONE;
}

constexpr bool none(BitChunk chunk) {
  return chunk == BITCHUNK_NONE;
}

constexpr std::size_t count(BitChunk chunk) {
  return __builtin_popcount(std::uintptr_t(chunk));
}

} // namespace omtalk

#endif // OMTALK_UTIL_BITCHUNK_H_
