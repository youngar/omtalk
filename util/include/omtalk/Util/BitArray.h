#ifndef OMTALK_GC_BITARRAY_HPP_
#define OMTALK_GC_BITARRAY_HPP_

#include <array>
#include <cstdint>
#include <omtalk/Util/Bytes.h>

namespace omtalk {

enum class BitChunk : std::uintptr_t {};

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

constexpr std::size_t BITCHUNK_NBITS = sizeof(BitChunk) * 8;

template <std::size_t N>
using BitChunkArray = std::array<BitChunk, N>;

template <std::size_t N>
class BitArray {
public:
  static_assert((N % BITCHUNK_NBITS) == 0,
                "BitArray size must be a multiple of BITCHUNK_NBITS.");

  static constexpr std::size_t NCHUNKS = N / BITCHUNK_NBITS;

  BitArray() = default;

  bool get(std::size_t index) const noexcept {
    return BitChunk(0) != (chunkForBit(index) & maskForBit(index));
  }

  bool set(std::size_t index) noexcept {
    if (!get(index)) {
      chunkForBit(index) |= maskForBit(index);
      return true;
    }
    return false;
  }

  bool unset(std::size_t index) noexcept {
    if (get(index)) {
      chunkForBit(index) &= ~maskForBit(index);
      return true;
    }
    return false;
  }

  std::size_t size() const noexcept { return chunks.size() * BITCHUNK_NBITS; }

  void clear() noexcept { chunks.fill(BitChunk(0)); }

private:
  static constexpr std::size_t indexForBit(std::size_t index) {
    return index / BITCHUNK_NBITS;
  }

  static constexpr std::size_t shiftForBit(std::size_t index) {
    return index % BITCHUNK_NBITS;
  }

  static constexpr BitChunk maskForBit(std::size_t index) {
    return BitChunk(1) << shiftForBit(index);
  }

  BitChunk &chunkForBit(std::size_t index) noexcept {
    return chunks.at(indexForBit(index));
  }

  const BitChunk &chunkForBit(std::size_t index) const noexcept {
    return chunks.at(indexForBit(index));
  }

  BitChunkArray<NCHUNKS> chunks;
};



} // namespace omtalk

#endif // OMTALK_GC_BITARRAY_HPP_
