#ifndef OMTALK_UTIL_BITCHUNK_H_
#define OMTALK_UTIL_BITCHUNK_H_

#include <array>
#include <cstdint>
#include <omtalk/Util/Bytes.h>

namespace omtalk {

enum class BitChunk : std::uintptr_t {};

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

} // namespace omtalk

#endif // OMTALK_UTIL_BITCHUNK_H_
